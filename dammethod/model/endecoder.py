import torch.nn as nn
from torchvision.models import resnet50

class Backbone(nn.Module):

    def __init__(self,pretrained=True):
        super(Backbone,self).__init__()

        org_resnet=resnet50(pretrained)

        self.conv1=nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=org_resnet.bn1
        self.relu=org_resnet.relu
        self.maxpool=org_resnet.maxpool
        self.layer1=org_resnet.layer1
        self.layer2=org_resnet.layer2
        self.layer3=org_resnet.layer3
        self.layer4=org_resnet.layer4

        self.avgpool=org_resnet.avgpool



    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # x=self.layer5(x)

        # x=self.avgpool(x)

        return x


class HeatmapHead(nn.Module):

    def __init__(self):
        super(HeatmapHead,self).__init__()

        self.relu=nn.ReLU(inplace=True)

        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self,scene_face_feat):

        # bs=x.size(0)

        # heatmap=self.scene_compress(x)
        # heatmap=self.scene_decoder(heatmap)

        # in_out=self.in_out_compress(x)
        # in_out=self.fc_inout(in_out.view(bs,49))

        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x #,in_out


class InoutDecoder(nn.Module):

    def __init__(self):
        super(InoutDecoder,self).__init__()

        self.relu=nn.ReLU(inplace=True)

        self.compress_conv1=nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=False)
        self.compress_bn1=nn.BatchNorm2d(256)
        self.compress_conv2=nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.compress_bn2=nn.BatchNorm2d(1)
        # self.compress_conv3=nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0,bias=False)
        # self.compress_bn3=nn.BatchNorm2d(1)

        self.fc_inout=nn.Linear(49,1)

    def forward(self,global_memory):

        x=self.compress_conv1(global_memory)
        x=self.compress_bn1(x)
        x=self.relu(x)

        x=self.compress_conv2(x)
        x=self.compress_bn2(x)
        x=self.relu(x)

        # x=self.compress_conv3(x)
        # x=self.compress_bn3(x)
        # x=self.relu(x)

        x=x.view(-1,49)
        inout=self.fc_inout(x)

        return inout

