from loss.loss_reconstruction import loss_reconstruction
from loss.loss_occupancy import loss_occupancy
from loss.loss_normal_consistency import loss_normal_consistency
from loss.loss_overlapping import loss_overlapping
from loss.loss_coverage import loss_coverage
from config import *

# losses are reconstruction loss, occupancy loss, normal consistency loss, overlapping loss, and convergence loss

#output is batch * 222 * 5 * 3

def loss_function(prediction, target, sum_loss):
    functions = [loss_reconstruction, loss_occupancy, loss_normal_consistency,
                loss_overlapping, loss_coverage]
    losses = []
    for w, f in zip(loss_weight, functions):
        if (w==0):
            continue
        losses.append(w * f(prediction, target, sum_loss))
    return sum(losses)

