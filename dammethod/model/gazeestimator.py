import torch
import torch.nn as nn
import torch.nn.functional as F
from gazeestimation.gazedirmodel import EyeFeatExtmodel, HeadPosExtmodel


class GazeDirmodel(nn.Module):

    def __init__(self):
        super(GazeDirmodel, self).__init__()

        # 3D gaze estimation
        self.leye_feat = EyeFeatExtmodel()
        self.reye_feat = EyeFeatExtmodel()
        self.head_pos = HeadPosExtmodel()

        self.fc1 = nn.Linear(1026, 256)
        self.fc2 = nn.Linear(256, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, head, l_eye, r_eye, indicator):
        # 3D gaze estimation
        l_eye_feat = self.leye_feat(l_eye)
        r_eye_feat = self.reye_feat(r_eye)

        eyes_feat = torch.cat([l_eye_feat, r_eye_feat], dim=1)


        eyes_feat = torch.mul(eyes_feat, indicator.unsqueeze(1))

        head_pos = self.head_pos(head)

        concat_feat = torch.cat([eyes_feat, head_pos], dim=1)

        concat_feat = self.fc1(concat_feat)
        concat_feat = self.relu(concat_feat)
        gaze_vector = self.fc2(concat_feat)


        norm_gazevector=F.normalize(gaze_vector,p=2,dim=1)
        # norm = torch.norm(gaze_vector, 2, dim=1)
        # norm_ = norm.clone()
        # norm_[norm_ <= 0] = 1.0
        # norm_gazevector = gaze_vector / norm_.view([-1, 1])

        return norm_gazevector