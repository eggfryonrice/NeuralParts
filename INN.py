import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix

class Invertible_Neural_Network(nn.Module):

    def __init__(self, n_feature, n_p_theta, n_layer, device, hidden=256, normalize=True, explicit_affine=True):
        super().__init__()
        self.n_feature = n_feature
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer
        self.split = [i%3 for i in range(n_layer)]#[random.choice([0, 1, 2]) for i in range(n_layer)] #
        self.device = device
        PPP = P_Theta_Layer(output_size=n_p_theta)

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(Conditional_Coupling_Layer(n_feature, n_p_theta, self.split[i], self.device, PPP))

        self.normalize = normalize
        self.explicit_affine = explicit_affine
        self.normalize_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.rotation_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.translation_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 3))
    
    def set_new_device(self, device):
        self.device = device
        for l in self.layers:
            l.device = device
            l.to(device)
        self.to(device)

    def forward(self, Cm, x):
        #print("got point ", x[0,0,0,:])
        y = x
        for i in range(self.n_layer):
            y = (self.layers)[i](Cm, y, 0)
        #print("before normalization and transform ", y[0,0,0,:])
        #print((nn.functional.elu(self.normalize_layer(Cm)) + 1)[:, None].shape)
        if (self.normalize):
            y = y / ((nn.functional.elu(self.normalize_layer(Cm)) + 1)[:, None])
        if (self.explicit_affine):

            rot_quart = self.rotation_layer(Cm)
            rot_quart_norm = rot_quart / torch.sqrt(torch.sum(torch.square(rot_quart), dim=-1, keepdim=True))
            rot_matx = quaternion_to_matrix(rot_quart_norm)
            trans_matx = torch.unsqueeze(self.translation_layer(Cm), dim=1).expand(-1, x.shape[1], -1, -1)
            u = torch.unsqueeze(rot_matx, dim=1)#.expand(-1, x.shape[1], -1, -1, -1)
            #print(u.shape)
            y = torch.matmul(y.unsqueeze(-2), u).squeeze(-2) + trans_matx
        return y
    
    def backward(self, Cm, x):
        y = x 
        if (self.explicit_affine):
            rot_quart = self.rotation_layer(Cm)
            rot_quart_norm = rot_quart / torch.sqrt(torch.sum(torch.square(rot_quart), dim=-1, keepdim=True))
            rot_matx = quaternion_to_matrix(rot_quart_norm)
            trans_matx = torch.unsqueeze(self.translation_layer(Cm), dim=1).expand(-1, x.shape[1], -1, -1)
            #print(self.translation_layer(Cm).shape)
            u = (y - trans_matx)
            v = torch.unsqueeze(rot_matx, dim=1)#.expand(-1, x.shape[1], -1, -1, -1).transpose(-2, -1)
            y = torch.matmul(u.unsqueeze(-2), v).squeeze(-2) 
        
        if (self.normalize):
            y = y * ((nn.functional.elu(self.normalize_layer(Cm)) + 1)[:, None])

        for i in range(self.n_layer):
            y = (self.layers)[self.n_layer -1 -i](Cm, y, 1)
        return y

class Conditional_Coupling_Layer(nn.Module):
    
    # Input :
    #   Cm : batch_size(4) X n_primitive(5) X size_Cm(512)
    #   Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
    #   point : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(3)
    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)
    #   inputpoint_split : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(1)

    def __init__(self, n_feature, n_p_theta, split, device, player, hidden=256):
        # split = 0 -> split x / split = 1 -> split y / split 2 -> split z
        super().__init__()
        self.split = split
        if (split == 0):
            self.nsplit = torch.tensor([0.,1.,1.]).float().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif (split == 1):
            self.nsplit = torch.tensor([1.,0.,1.]).float().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif (split == 2):
            self.nsplit = torch.tensor([1.,1.,0.]).float().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError
        self.ptheta_layer = player #P_Theta_Layer(output_size=n_p_theta)
        self.stheta_layer = S_Theta_Layer(n_feature+n_p_theta, hidden=hidden)
        self.ttheta_layer = T_Theta_Layer(n_feature+n_p_theta, hidden=hidden)
        self.device = device
        self.ptheta_layer.to(self.device)
        self.stheta_layer.to(self.device)
        self.ttheta_layer.to(self.device)

    def forward(self, Cm, point, inv):#
        #print("coupling layer with split ", self.split, "got point ", point[0,0,0,:])
        B, N, M, D = point.shape
        assert (D == 3)
        point_nsplit = self.nsplit * point 
        point_split = (((1-self.nsplit) * point)[:, :, :, self.split]).unsqueeze(-1)
        #print("point_split size is : " + str(point_split.shape))
        if inv :
            newpoint_split = self.backward_sub(Cm, point_nsplit, point_split)
        else :
            newpoint_split = self.forward_sub(Cm, point_nsplit, point_split)

        #print("newpoint_split size is : " + str(newpoint_split.shape))
        result = point_nsplit + (1 - self.nsplit) * (newpoint_split.expand(-1, -1, -1, 3))
        
        #print("result of coupling layer ", result[0,0,0,:])
        return result
    
    def backward_sub(self, Cm, inputpoint_nsplit, inputpoint):
        x = self.ptheta_layer(inputpoint_nsplit)
        s = self.stheta_layer(Cm, x)
        t = self.ttheta_layer(Cm, x)
        #print("backward : s size is : " + str(s.shape))
        #print("backward : t size is : " + str(t.shape))
        #print("backward : inputpoint size is : " + str(inputpoint.shape))
        outputpoint_split = t + (inputpoint * torch.exp(s))
        #print("backward : outputpoint size is : " + str(outputpoint_split.shape))
        return outputpoint_split 

    def forward_sub(self, Cm, outputpoint_nsplit, outputpoint):
        x = self.ptheta_layer(outputpoint_nsplit)
        s = self.stheta_layer(Cm, x)
        t = self.ttheta_layer(Cm, x)
        #print("forward : s size is : " + str(s.shape))
        #print("forward : t size is : " + str(t.shape))
        #print("forward : inputpoint size is : " + str(outputpoint.shape))
        inputpoint_split = (outputpoint - t) * torch.exp((-1) * s)
        #print("forward : outputpoint size is : " + str(inputpoint_split.shape))

        return inputpoint_split
        


class P_Theta_Layer(nn.Module):

    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)

    def __init__(self, output_size=128):
        hidden = 2 * output_size
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.layer2 = nn.Linear(hidden, output_size)
    
    def forward(self, inputpoint_nsplit):
        x = self.layer1(inputpoint_nsplit)
        return self.layer2(x)

class S_Theta_Layer(nn.Module):

    def __init__(self, input_dim, hidden=256):
        hidden1, hidden2, hidden3 = hidden, hidden, 1
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden2, hidden3), nn.Hardtanh(min_val = -10, max_val = 10))
        self.input_dim = input_dim
    
    def forward(self, Cm, ptheta_result):
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, ptheta_result.shape[1], -1, -1)
        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        #print("Size of x is " + str(x.shape) + " , input_dim is " + str(self.input_dim))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print("Size of final x is " + str(x.shape))

        return x

class T_Theta_Layer(nn.Module):

    def __init__(self, input_dim, hidden=256):
        hidden1, hidden2, hidden3 = hidden, hidden, 1
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.layer3 = nn.Linear(hidden2, hidden3)
    
    def forward(self, Cm, ptheta_result):
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, ptheta_result.shape[1], -1, -1)
        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
