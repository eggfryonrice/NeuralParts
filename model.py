import torch
import torch.nn as nn
from torchvision import models
import feature_extractor as FE
import INN as INN_mod
import utils.get_points_from_sphere as gs
import random
from config import *

def Model(device):
    return Model_overall(512,n_primitive, 200, 128, 4, device)

class Model_overall(nn.Module):
    def __init__(self, n_feature, n_primitive, n_points, n_p_theta, n_layer, device):
        super().__init__()
        self.n_feature = n_feature
        self.n_primitive = n_primitive
        self.n_points = n_points
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer
        self.device = device

        self.fe = FE.FeatureExtractor(n_feature//2, n_primitive)
        self.INN = INN_mod.Invertible_Neural_Network(n_feature, n_p_theta, n_layer, device)

    def set_new_device(self,device):
        self.device = device
        self.INN.set_new_device(device)
        self.to(device)

    def forward(self, x):
        image = x[0]
        surface_samples = x[1]
        volume_samples = x[2]

        Cm = self.fe(image)
        Batch = Cm.shape[0]
        inputpoint = gs.fx_sample_sphere(Batch, 2 * 60 ** 2 - 2 * 60 + 2, self.n_primitive, randperm=False) * sphere_radius
        inputpoint = inputpoint[:,torch.randperm(inputpoint.shape[1]),:,:]
        inputpoint = inputpoint[:,:self.n_points,:,:]
        #inputpoint = gs.fx_sample_sphere(Batch, 2 * 11 ** 2 - 2 * 11 + 2, self.n_primitive, randperm=False) * sphere_radius
        inputpoint = inputpoint.to(self.device)

        points_primitives = self.INN(Cm, inputpoint)

        points_volume_expanded = volume_samples[:,:,:3].unsqueeze(2).expand(-1,-1,5,-1)
        y_volume = self.INN.backward(Cm, points_volume_expanded)
        g_m_volume = y_volume.pow(2).sum(3).pow(0.5) - sphere_radius

        points_surface = surface_samples[:,:,:3]
        points_surface.requires_grad_()
        points_surface_expanded = points_surface.unsqueeze(2).expand(-1,-1,5,-1)
        y_surface = self.INN.backward(Cm, points_surface_expanded)
        g_m_surface = y_surface.pow(2).sum(-1).pow(0.5) - sphere_radius
        G_surface = g_m_surface.min(-1)[0]
        gradient_G_surface = torch.autograd.grad(G_surface.sum(), points_surface, retain_graph=True, create_graph=True)[0]

        assert(points_primitives.requires_grad)
        assert(g_m_volume.requires_grad)
        assert(gradient_G_surface.requires_grad)

        return [points_primitives, g_m_volume, gradient_G_surface]


    def primitive_points(self, x, n):
        image = x[0]
        surface_samples = x[1]
        volume_samples = x[2]

        Cm = self.fe(image)
        Batch = Cm.shape[0]
        inputpoint = gs.fx_sample_sphere(Batch, n, self.n_primitive, randperm=False) * sphere_radius
        inputpoint = inputpoint.to(self.device)

        points_primitives = self.INN(Cm, inputpoint)

        return points_primitives

    def eval(self, x):
        image = x[0]
        surface_samples = x[1]
        volume_samples = x[2]

        Cm = self.fe(image)
        Batch = Cm.shape[0]
        inputpoint = gs.fx_sample_sphere(Batch, self.n_points, self.n_primitive, randperm=False)
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, inputpoint.shape[1], -1, -1)



    
    def backward(self, Cm_ext, outputpoint):
        # Cm_ext : tensor of size batch_size X n_points X n_primitive X n_feature
        # outputpoint : tensor of size batch_size X n_points X n_primitive X 3
        # Output : tensor of size batch_size X n_points X n_primitive X 3
        return self.INN.backward(Cm_ext, outputpoint)

