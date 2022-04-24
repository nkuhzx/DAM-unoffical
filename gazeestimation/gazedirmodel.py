import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34
import math

class EyeFeatExtmodel(nn.Module):

    def __init__(self):
        super(EyeFeatExtmodel,self).__init__()

        fullresnet=resnet18(pretrained=True)

        self.feat_extract = nn.Sequential(*list(fullresnet.children())[0:-1])


    def forward(self,eye_img):

        feat=self.feat_extract(eye_img)
        feat=torch.flatten(feat,1)
        return feat

class HeadPosExtmodel(nn.Module):

    def __init__(self):
        super(HeadPosExtmodel,self).__init__()

        fullresnet=resnet34(pretrained=True)


        self.feat_extract = nn.Sequential(*list(fullresnet.children())[0:-1])

        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,2)
        self.LeakyRelU=nn.LeakyReLU()
        # fc1=nn.Linear(self.feat_extract.fc.in_features,)


    def forward(self,eye_img):

        feat=self.feat_extract(eye_img)
        feat=torch.flatten(feat,1)
        feat=self.fc1(feat)
        feat=self.LeakyRelU(feat)
        feat=self.fc2(feat)

        angular_output=feat[:,:2]
        angular_output[:,0]=math.pi*nn.Tanh()(angular_output[:,0])
        angular_output[:,1]=(math.pi/2)*nn.Tanh()(angular_output[:,1])

        return angular_output


class GazeDirmodel(nn.Module):

    def __init__(self):
        super(GazeDirmodel,self).__init__()

        self.leye_feat=EyeFeatExtmodel()
        self.reye_feat=EyeFeatExtmodel()
        self.head_pos=HeadPosExtmodel()

        self.fc1=nn.Linear(1026,256)
        self.fc2=nn.Linear(256,3)
        self.relu=nn.ReLU(inplace=True)


    def forward(self,head,l_eye,r_eye,indicator):

        l_eye_feat=self.leye_feat(l_eye)
        r_eye_feat=self.reye_feat(r_eye)

        eyes_feat=torch.cat([l_eye_feat,r_eye_feat],dim=1)

        eyes_feat=torch.mul(eyes_feat,indicator)

        head_pos=self.head_pos(head)

        concat_feat=torch.cat([eyes_feat,head_pos],dim=1)

        concat_feat=self.fc1(concat_feat)
        concat_feat=self.relu(concat_feat)
        gaze_vector=self.fc2(concat_feat)

        norm = torch.norm(gaze_vector, 2, dim=1)
        norm_=norm.clone()
        norm_[norm_<=0]=1.0
        norm_gazevector=gaze_vector/norm_.view([-1,1])

        return {"gaze_vector":norm_gazevector,
                "head_pos":head_pos}



if __name__ == '__main__':

    fake_leye=torch.zeros((4,3,36,60))
    fake_reye=torch.zeros((4,3,36,60))

    fake_head=torch.zeros((4,3,224,224))

    fake_indicator=torch.zeros((4,1))

    gaze_model=GazeDirmodel()

    gaze_model(fake_head,fake_leye,fake_reye,fake_indicator)