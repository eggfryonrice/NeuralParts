import torch
from dataloader.dataloader import build_dataloader
from model import Model
from loss.loss_function import loss_function
from metric.metric_function import metric_function
import os, sys
from config import *

def main():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
    model = Model(device).to(device)
    train_dataloader = build_dataloader('dfaust',['train'])
    val_dataloader = build_dataloader('dfaust',['val'])
    loss_fn = loss_function
    metric_fn = metric_function
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer.zero_grad()
    best_validation = 0

    for epoch in range(n_epoch):
        aggregate = 0
        sum_loss = [0.,0.,0.,0.,0.]
        sum_metric = [0.,0.]
        for b, target in zip(list(range(n_step_per_epoch)), train_dataloader.infinite_iterator()):
            aggregate += 1
            for i in range(len(target)):
                # print(target[i].shape)
                target[i] = target[i].to(device)
            prediction = model(target)
            loss = loss_fn(prediction, target, sum_loss)
            metric_fn(prediction, target, sum_metric)
            #print("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            sys.stdout.write("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            loss.backward()
            if (aggregate == n_aggregate):
                optimizer.step()
                aggregate = 0
                optimizer.zero_grad()
        print("epoch %d, batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (epoch+1, b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
        torch.save(model.state_dict(), "model.pth")
        if (epoch+1) % 10 == 0:
            print("--------------validation--------------\n")
            sum_loss = [0.,0.,0.,0.,0.]
            sum_metric = [0.,0.]
            
            for b, target in zip(list(range(len(val_dataloader))), val_dataloader):
                for i in range(len(target)):
                    target[i] = target[i].to(device)
                prediction = model(target)
                loss = loss_fn(prediction, target, sum_loss)
                loss.backward()
                metric_fn(prediction, target, sum_metric)
                sys.stdout.write("batch: %d, losses: %.5f, %.5f, %.5f, %.5f, %.5f, iou: %.5f, chamferL1: %.5f \r" % (b+1, sum_loss[0]/(b+1), sum_loss[1]/(b+1), sum_loss[2]/(b+1), sum_loss[3]/(b+1), sum_loss[4]/(b+1), sum_metric[0]/(b+1), sum_metric[1]/(b+1)))
            optimizer.zero_grad()
            print("\n")
            if sum_metric[0] > best_validation:
                best_validation = sum_metric[0]
                torch.save(model.state_dict(), "best_model.pth")
                print("model saved at iou %.5f as best_model.pth"%(sum_metric[0]/(b+1)))

            print("--------------------------------------")
    

if __name__=='__main__':
    main()
