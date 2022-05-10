import torch
import torch.nn as nn
from dammethod.model.gazeestimator import GazeDirmodel
from dammethod.model.endecoder import Backbone,HeatmapHead,InoutDecoder
import math
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DAMmodel(nn.Module):

    def __init__(self,pretrained=True,inout_branch=False):
        super(DAMmodel,self).__init__()

        self.inout_branch=inout_branch

        self.gaze_estimator=GazeDirmodel()

        self.relu=nn.ReLU(inplace=True)

        self.alpha=6
        self.sigma=0.3

        self.backbone=Backbone(pretrained)

        self.heatmap_decoder=HeatmapHead()

        if self.inout_branch:
            self.inout_decoder=InoutDecoder()
        #
        # if inout_branch


    def forward(self,img,mimg,gazefield,head,l_eye,r_eye,indicator):


        bs=img.shape[0]

        Mfront=mimg[:,0:1,:,:]
        Mmid=mimg[:,1:2,:,:]
        Mback=mimg[:,2:,:]

        gazevector=self.gaze_estimator(head,l_eye,r_eye,indicator)
        # print(torch.isnan(gazevector).any())
        # print(gazevector)

        gaze_xy=gazevector[:,0:2]

        # norm=torch.norm(gaze_xy,2 ,dim=1)
        #
        # norm_=norm.clone()
        # norm_[norm_<=0]=1.0

        norm_gaze_xy=F.normalize(gaze_xy,p=2,dim=1)
            #gaze_xy/norm_.view([-1,1])


        gazefield=gazefield.permute([0,2,3,1]).reshape(-1,224*224,2)


        theta_M=torch.matmul(gazefield,norm_gaze_xy.unsqueeze(2)).detach()



        # print(theta_M.shape)
        # print(theta_M.min(),theta_M.max())
        theta_M=torch.clip(theta_M,-1,1)
        theta_M=theta_M.reshape(-1,224,224,1)


        theta_M=theta_M.permute([0,3,1,2])
        # theta_M_show=theta_M[0,0,:,:]

        theta_M=torch.arccos(theta_M)


        Mf=1-(self.alpha*theta_M/math.pi)
        Mf=self.relu(Mf)


        gaze_z=gazevector[:,2]

        Md=torch.zeros((bs,1,224,224))
        Md=Md.to(mimg.device)

        choice_1=(gaze_z>=-1) & (gaze_z<=-self.sigma)
        Md[choice_1]=Mfront[choice_1]

        choice_2=(gaze_z>-self.sigma) & (gaze_z<=self.sigma)
        Md[choice_2]=Mmid[choice_2]

        choice_3=(gaze_z>self.sigma) & (gaze_z<=1)
        Md[choice_3]=Mback[choice_3]

        Mdual=torch.mul(Md,Mf)

        # Md_show=Md[0,0,:,:]
        #
        # Mf_show=Mf[0,0,:,:]

        # print(Mdual.shape)
        # Mdual_show=Mdual[0,0,:,:]

        # Md_show=Md_show.detach().cpu().numpy()
        # Mf_show=Mf_show.detach().cpu().numpy()
        #
        # theta_M_show=theta_M_show.detach().cpu().numpy()
        #
        # Mdual_show=Mdual[0,0,:,:]
        #
        # Mdual_show = Mdual_show.detach().cpu().numpy()
        #
        # plt.imshow(Mf_show,cmap='jet')
        # plt.show()


        # print('thetaM',torch.isnan(Mf).any(),torch.isnan(Md).any(),torch.isnan(mimg).any())

        concat_img=torch.cat([img,Mdual],dim=1)


        shared_feat=self.backbone(concat_img)

        heatmap=self.heatmap_decoder(shared_feat)

        if self.inout_branch:

            inout=self.inout_decoder(shared_feat)
        else:
            inout=None


        outs = {
            'heatmap': heatmap,
            'gazevector':norm_gaze_xy,
            'inout':inout,
            'fov_att':Mf,
            "depth_att":Md
        }

        return outs



if __name__ == '__main__':

    # gazedirmodel=GazeDirmodel()
    #
    # print(gazedirmodel)
    #
    # checkpoint=torch.load("/home/hzx/project/DAM_process/gazeestimation/checkpoints/gaze360_5epoch.pth.tar")
    #
    # gazedirmodel.load_state_dict(checkpoint['state_dict'])
    #
    # torch.save(gazedirmodel.state_dict(), "../modelparas/gazedirmodel.pt")

    bs=4

    img=torch.randn((bs,3,224,224))

    dimg=torch.randn((bs,3,224,224))

    gazefield=torch.randn((bs,224,224,2))

    gazefield_norm=torch.norm(gazefield,2,dim=3)

    gazefield=gazefield/gazefield_norm.view([-1,224,224,1])
    print(gazefield.shape)

    headimg=torch.randn((bs,3,224,224))

    l_eye=torch.randn((bs,3,36,60))

    r_eye=torch.randn((bs,3,36,60))

    ind=torch.randn(bs,1)

    ind.bool()



    model=DAMmodel(inout_branch=True)

    model(img,dimg,gazefield,headimg,l_eye,r_eye,ind)