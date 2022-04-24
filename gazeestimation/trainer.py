import numpy as np


from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn

from gazeestimation.utils.utils import AverageMeter,MovingAverageMeter,Angularto3d_matrix



class Trainer(object):

    def __init__(self,model,criterion,optimizer,trainloader,valloader,opt,writer=None):
        super(Trainer,self).__init__()

        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer

        self.coslosses=MovingAverageMeter(100)
        self.anglosses=MovingAverageMeter(100)
        self.totallosses=MovingAverageMeter(100)

        self.evallosses=AverageMeter()

        self.trainloader= trainloader
        self.valloader=valloader


        self.device=opt.device

        self.opt=opt

    def train(self,epoch):

        self.model.train()

        # reset the loss value
        self.coslosses.reset()

        loader_capacity=len(self.trainloader)
        pbar=tqdm(total=loader_capacity)
        for i,data in enumerate(self.trainloader,0):

            self.optimizer.zero_grad()

            x_imgs,x_l_eye,x_r_eye,x_ind,gaze_angular,gaze_vector=data

            x_imgs=x_imgs.to(self.device)
            x_l_eye=x_l_eye.to(self.device)
            x_r_eye=x_r_eye.to(self.device)
            x_ind=x_ind.to(self.device)

            bs=x_imgs.shape[0]


            y_gaze_angular=gaze_angular.to(self.device)
            y_gaze_vector=gaze_vector.to(self.device)

            output=self.model(x_imgs,x_l_eye,x_r_eye,x_ind)

            pred_gaze_angular=output["head_pos"]
            pred_gaze_vector=output["gaze_vector"]

            gaze_angular_loss=self.criterion[0][0](pred_gaze_angular,y_gaze_angular)

            gaze_vector_loss=self.criterion[0][1](pred_gaze_vector,y_gaze_vector)
            gaze_vector_loss=1-torch.sum(gaze_vector_loss)/bs


            total_loss=gaze_angular_loss*10000+gaze_vector_loss*1000

            total_loss.backward()
            self.optimizer.step()

            self.coslosses.update(gaze_vector_loss.detach().cpu().numpy())
            self.anglosses.update(gaze_angular_loss.detach().cpu().numpy())
            self.totallosses.update(total_loss.detach().cpu().numpy())

            if i%self.opt.evalrec==0 and i>0:

                self.valid(self.opt,epoch)


            pbar.set_description("Epoch: [{0}]".format(epoch))
            pbar.set_postfix(cosloss=self.coslosses.avg,
                             angloss=self.anglosses.avg,
                             total_loss=self.totallosses.avg,
                             evalinval=self.evallosses.avg,
                             lr=self.optimizer.param_groups[0]["lr"])
            pbar.update(1)

        pbar.close()

    @torch.no_grad()
    def valid(self,opt,epoch):

        self.model.eval()

        self.evallosses.reset()
        for i, data in enumerate(self.valloader,0):
            x_imgs,x_l_eye,x_r_eye,x_ind,gaze_angular,gaze_vector=data

            x_imgs=x_imgs.to(self.device)
            x_l_eye=x_l_eye.to(self.device)
            x_r_eye=x_r_eye.to(self.device)
            x_ind=x_ind.to(self.device)

            bs=x_imgs.shape[0]

            y_gaze_vector=gaze_vector.to(self.device)

            output=self.model(x_imgs,x_l_eye,x_r_eye,x_ind)

            pred_gaze_vector=output["gaze_vector"]
            # pred_gaze_vector=pred_gaze_vector.to(self.device)
            # print(pred_gaze_vector.device,y_gaze_vector.device)
            eval_value=self.criterion[1](pred_gaze_vector,y_gaze_vector)
            eval_value=torch.sum(eval_value)/bs

            self.evallosses.update(eval_value.detach().cpu().numpy())

        self.model.train()