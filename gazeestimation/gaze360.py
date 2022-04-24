import os

import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F


from PIL import Image

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

import matplotlib.pyplot as plt


class Gaze360Loader(object):

    def __init__(self, opt):
        self.train_dataset = Gaze360Dataset(opt.root_dir, opt.train_csv,'train', opt, show=False)
        self.val_dataset = Gaze360Dataset(opt.root_dir, opt.val_csv,'val', opt, show=False)
        self.test_dataset = Gaze360Dataset(opt.root_dir, opt.test_csv,'test', opt, show=False)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=opt.train_batch_size,
                                       num_workers=opt.num_worker,
                                       shuffle=True)

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.val_batch_size,
                                     num_workers=opt.num_worker,
                                     shuffle=False)

        self.test_loader = DataLoader(self.test_dataset,
                                       batch_size=opt.test_batch_size,
                                       num_workers=opt.num_worker,
                                       shuffle=False)


class Gaze360Dataset(Dataset):

    def __init__(self,dataset_dir,csv_path,type,opt=None,show=False):

        self.dataset_dir=dataset_dir
        self.csv_path=csv_path
        self.opt=opt

        self.pd=pd.read_csv(csv_path,header=None)

        self.input_size=224

        transform_list=[]
        transform_list.append(transforms.Resize((self.input_size,self.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))

        self.transform=transforms.Compose(transform_list)

        transform_list=[]
        transform_list.append(transforms.Resize((36,60)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))

        self.eye_transform=transforms.Compose(transform_list)

        self.show=show

        self.train=True if type=="train" else False


    def __getitem__(self, i):

        img_path=os.path.join(self.dataset_dir,self.pd.iloc[i,0])

        gaze_vector=np.array([self.pd.iloc[i,1],self.pd.iloc[i,2],self.pd.iloc[i,3]],dtype=np.float64)

        left_eye_box=np.array(self.pd.iloc[i,4:8])

        right_eye_box=np.array(self.pd.iloc[i,8:12])

        img=Image.open(img_path)
        img=img.convert('RGB')

        img_W,img_H=img.size

        # left eye box
        if -1 in left_eye_box:
            left_eye_box=np.array([False,False,False,False])
        else:
            left_eye_box[0: : 2]=left_eye_box[0: : 2]*img_W
            left_eye_box[1: :2]=left_eye_box[1: :2]*img_H
            left_eye_box[2:]=left_eye_box[2:]+left_eye_box[0:2]

        # right eye box
        if -1 in right_eye_box:
            right_eye_box=np.array([False,False,False,False])
        else:
            right_eye_box[0: : 2]=right_eye_box[0: : 2]*img_W
            right_eye_box[1: :2]=right_eye_box[1: :2]*img_H
            right_eye_box[2:]=right_eye_box[2:]+right_eye_box[0:2]


        # Data augument for training
        if self.train:

            # Random flip
            if True:
            # if np.random.random_sample()<=0.5:
                img=img.transpose(Image.FLIP_LEFT_RIGHT)
                gaze_vector[0]=-gaze_vector[0]

                # if False not in le
                if left_eye_box.sum()!=False:
                    left_eye_box[0::2]=img_W-left_eye_box[0::2]
                    left_eye_box[0], left_eye_box[2] = left_eye_box[2], left_eye_box[0]


                if right_eye_box.sum()!=False:
                    right_eye_box[0::2]=img_W-right_eye_box[0::2]
                    right_eye_box[0],right_eye_box[2]=right_eye_box[2],right_eye_box[0]

                    # print(right_eye_box)

                left_eye_box,right_eye_box=right_eye_box,left_eye_box


            # Random color change
            if np.random.random_sample()<=0.5:

                img=TF.adjust_brightness(img,brightness_factor=np.random.uniform(0.5,1.5))
                img=TF.adjust_contrast(img,contrast_factor=np.random.uniform(0.5,1.5))
                img=TF.adjust_saturation(img,saturation_factor=np.random.uniform(0.5,1.5))

        if left_eye_box.sum()!=False:
            left_eye_img=img.crop(tuple(left_eye_box))

        else:
            left_eye_img=None

        if right_eye_box.sum()!=False:
            right_eye_img=img.crop(tuple(right_eye_box))
        else:
            right_eye_img=None

        if left_eye_box.sum()==False or right_eye_box.sum()==False:

            indicator=False
        else:
            indicator=True

        # print(left_eye_img.size)
        # fig,ax=plt.subplots(2,2)
        # ax[0][0].imshow(img)
        # ax[1][0].imshow(left_eye_img)
        # ax[1][1].imshow(right_eye_img)
        # plt.show()
        # for show
        if self.show:
            import cv2
            # img_show=img
            img_show=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
            left_eye_box=left_eye_box.astype(int)
            right_eye_box=right_eye_box.astype(int)

            cv2.rectangle(img_show,(left_eye_box[0],left_eye_box[1]),
                                    (left_eye_box[2],left_eye_box[3]),(255,0,0),2)

            cv2.rectangle(img_show,(right_eye_box[0],right_eye_box[1]),
                                    (right_eye_box[2],right_eye_box[3]),(0,255,0),2)

            cv2.imshow('eye',img_show)

            cv2.waitKey(0)

        if self.transform is not None:
            img=self.transform(img)
        if self.eye_transform is not None:
            if left_eye_img is not None:
                left_eye_img=self.eye_transform(left_eye_img)
            else:
                left_eye_img=torch.zeros((3,36,60))

            if right_eye_img is not None:
                right_eye_img=self.eye_transform(right_eye_img)
            else:
                right_eye_img=torch.zeros((3,36,60))

        indicator=torch.FloatTensor([indicator])

        gaze_vector=torch.FloatTensor(gaze_vector)
        normalized_gaze=F.normalize(gaze_vector.view(1,3)).view(3)

        spherical_vector=torch.FloatTensor(2)
        spherical_vector[0] = math.atan2(normalized_gaze[0],-normalized_gaze[2])
        spherical_vector[1] = math.asin(normalized_gaze[1])

        # spherical_vector[0] = math.atan2(normalized_gaze[0],-normalized_gaze[2])
        # spherical_vector[1] = math.asin(normalized_gaze[1])

        return img,left_eye_img,right_eye_img,indicator,spherical_vector,normalized_gaze

    def __len__(self):

        return len(self.pd)


if __name__ == '__main__':

    # test for dataset

    train_dataset=Gaze360Dataset("/home/nku120/HZX/dataset/Gaze360/imgs",
                                 "/gazeestimation/tools/train_eye.txt",
                                 type="train",
                                 show=True)

    for i in range(len(train_dataset)):

        train_dataset.__getitem__(i)


    # train_loader = DataLoader(train_dataset,
    #                                    batch_size=64,
    #                                    num_workers=6,
    #                                    shuffle=True)
    #
    # pbar=tqdm(total=len(train_loader))
    # for i,data in enumerate(train_loader,0):
    #
    #     x_imgs,x_l_eye,x_r_eye, gaze_angular, _ = data
    #     # print(x_imgs.shape)
    #     # print(x_l_eye.shape)
    #     # print(x_r_eye.shape)
    #     pbar.update(1)
    #
    # pbar.close()












