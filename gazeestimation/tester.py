import numpy as np


from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn

from gazeestimation.utils.utils import AverageMeter,Angularto3d_matrix


class Tester(object):

    def __init__(self,model,criterion,testloader,opt,writer=None):


        self.model=model
        self.criterion=criterion

        self.testloader=testloader

        self.evallosses=AverageMeter()

        self.device=opt.device

        self.opt=opt

    @torch.no_grad()
    def test(self,epoch,best_error):

        self.model.eval()

        self.evallosses.reset()

        loader_capacity=len(self.testloader)
        pbar=tqdm(total=loader_capacity)
        for i,data in enumerate(self.testloader,0):
            x_imgs,x_l_eye,x_r_eye,x_ind,gaze_angular,gaze_vector=data

            x_imgs=x_imgs.to(self.device)
            x_l_eye=x_l_eye.to(self.device)
            x_r_eye=x_r_eye.to(self.device)
            x_ind=x_ind.to(self.device)

            bs=x_imgs.shape[0]

            y_gaze_vector=gaze_vector.to(self.device)

            output=self.model(x_imgs,x_l_eye,x_r_eye,x_ind)

            # pred_gaze_vector=Angularto3d_matrix(pred_gaze_angular)
            # pred_gaze_vector=pred_gaze_vector.to(self.device)
            pred_gaze_vector=output["gaze_vector"]

            eval_value=self.criterion[1](pred_gaze_vector,y_gaze_vector)
            eval_value=torch.sum(eval_value)/bs

            self.evallosses.update(eval_value.detach().cpu().numpy())

            pbar.set_description("Test After Epoch: [{0}]".format(epoch))
            pbar.set_postfix(evalinval=self.evallosses.avg)

            pbar.update(1)

        pbar.close()

        return self.evallosses.avg
