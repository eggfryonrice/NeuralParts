from config import *
import torch

def loss_occupancy(prediction, target, sum_loss):
    
    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] # batch * N
    t_w = target[2][:,:,4] # batch * N

    g_m = prediction[1] # batch * N * M

    G = g_m.min(-1)[0] # batch * N
    p_classification = -1 * G / temperature
    # BCELoss = torch.nn.BCEWithLogitsLoss(weight = t_w, reduction="mean")
    # loss = BCELoss(p_classification, t_labels)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        p_classification, t_labels, reduction="none"
    )
    loss = t_w * loss
    loss = loss.mean()

    sum_loss[1] += loss.item()
    assert(loss.requires_grad)
    return loss