import torch
import trimesh
from model import Model, Model_overall
from dataloader.dataloader import build_dataloader
from loss.loss_function import loss_function
from metric.metric_function import metric_function
import utils.get_points_from_sphere as gs
from dataloader.dataset import build_dataset
import os, sys
import matplotlib.pyplot as plt

def get_overall_mesh():
    device =  torch.device('cpu')
    model = Model(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.set_new_device(device)
    #model = Model(device)
    target = build_dataset('dfaust', ['train']).get(6000) #6000
    sampled_surface = target[1][0]
    plt.scatter(sampled_surface[:,0], sampled_surface[:,1], s=1)
    plt.savefig("surface_sample_plot.png")
    for i in range(len(target)):
        target[i] = target[i].to(device)
    num_points = 40 ** 2 - 40 * 2 + 2
    prediction = model.primitive_points(target, num_points)
    print(prediction.shape)
    p_p = prediction
    B, n_points, n_primitives, D = p_p.shape
    assert (D == 3)
    face = (gs.fx_sample_face(B, num_points, n_points, d=3, randperm=False))
    #print("face shape is : " + str(len(face)) + " * " + str(len(face[0])))
    #print("face is : " + str(face))
    #print("number of primitives : " + str(n_primitives))
    Listmesh = []
    for j in range(n_primitives):
        c = p_p[0, :, j, :].cpu().detach().numpy()
        #print("shape of c is : " + str(len(c)) + " * " + str(len(c[0])))
        Listmesh.append(trimesh.Trimesh(c, face))
    trimesh.util.concatenate(Listmesh).export(file_obj=("./mesh_overall.obj"), file_type='obj')

get_overall_mesh()
