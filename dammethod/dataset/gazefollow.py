import numpy as np
import pandas as pd
import os
import math

from PIL import Image, ImageFilter, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# from vsgia_model.models.utils.misc import nested_tensor_from_tensor_list
from dammethod.utils import img_utils


import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


class GazeFollowLoader(object):

    def __init__(self,opt):

        self.train_gaze = GazeFollowDataset(opt.DATASET.train_anno, 'train', opt, show=False)
        self.val_gaze = GazeFollowDataset(opt.DATASET.test_anno, 'test', opt, show=False)


        self.train_loader=DataLoader(self.train_gaze,
                                     batch_size=opt.DATASET.train_batch_size,
                                     num_workers=opt.DATASET.load_workers,
                                     shuffle=True,
                                     collate_fn=collate_fn)


        self.val_loader=DataLoader(self.val_gaze,
                                   batch_size=opt.DATASET.test_batch_size,
                                   num_workers=opt.DATASET.load_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)

class GazeFollowDataset(Dataset):

    def __init__(self, csv_path, type, opt,show=False):

        test=True if type=='test' else False

        if test:
            df = pd.read_csv(csv_path, sep=',', index_col=False, encoding="utf-8-sig")

            df = df[['rgb_path','depth_path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max',
                    'head_bbox_y_max','l_eye_xmin','l_eye_ymin','l_eye_xmax','l_eye_ymax','r_eye_xmin','r_eye_ymin','r_eye_xmax','r_eye_ymax']].groupby(['rgb_path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)



        else:
            df = pd.read_csv(csv_path, sep=',', index_col=False, encoding="utf-8-sig")
            df = df[df['in_or_out'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)


            df.reset_index(inplace=True)
            self.y_train = df[['gaze_x','gaze_y', 'in_or_out','id']]
            self.X_train = df[['rgb_path','depth_path','head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max','eye_x', 'eye_y',
                               'l_eye_xmin','l_eye_ymin','l_eye_xmax','l_eye_ymax','r_eye_xmin','r_eye_ymin','r_eye_xmax','r_eye_ymax']]
            self.length = len(df)

        self.data_dir = opt.DATASET.root_dir
        self.depth_dir=opt.DATASET.depth_dir

        transform_list = []
        transform_list.append(transforms.Resize((opt.TRAIN.input_size, opt.TRAIN.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize((opt.TRAIN.eye_size[0], opt.TRAIN.eye_size[1])))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.eye_transform=transforms.Compose(transform_list)

        self.depth_resize=(224,224)

        self.test = test

        self.input_size = opt.TRAIN.input_size
        self.output_size = opt.TRAIN.output_size
        self.imshow = show

        self.tau=16


    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['rgb_path']
                depth_path=row['depth_path']
                x_min = row['head_bbox_x_min']
                y_min = row['head_bbox_y_min']
                x_max = row['head_bbox_x_max']
                y_max = row['head_bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                leye_x_min=row['l_eye_xmin']
                leye_y_min=row['l_eye_ymin']
                leye_x_max=row['l_eye_xmax']
                leye_y_max=row['l_eye_ymax']
                reye_x_min=row['r_eye_xmin']
                reye_y_min=row['r_eye_ymin']
                reye_x_max=row['r_eye_xmax']
                reye_y_max=row['r_eye_ymax']

                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up

            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside


        else:
            path,depth_path,x_min, y_min, x_max, y_max, eye_x, eye_y, \
            leye_x_min,leye_y_min,leye_x_max,leye_y_max,reye_x_min,reye_y_min,reye_x_max,reye_y_max = self.X_train.iloc[index]

            gaze_x, gaze_y, inout, graph_id = self.y_train.iloc[index]

            gaze_inside = bool(inout)



        # crop the eyeimg
        head_loc=np.array([x_min, y_min, x_max, y_max]).astype(int)
        head_loc[head_loc<0]=0
        if head_loc[1]>head_loc[3]:
            head_loc[3],head_loc[1]=head_loc[1],head_loc[3]
        if head_loc[0]>head_loc[2]:
            head_loc[0],head_loc[2]=head_loc[2],head_loc[0]


        if leye_x_min+leye_y_min+leye_x_max+leye_y_max<0:
            leye_loc=np.array([-1,-1,-1,-1])
        else:
            leye_loc=np.array([leye_x_min+head_loc[0],leye_y_min+head_loc[1],leye_x_max+head_loc[0],leye_y_max+head_loc[1]])
        if reye_x_min+reye_y_min+reye_x_max+reye_y_max<0:
            reye_loc=np.array([-1,-1,-1,-1])
        else:
            reye_loc=np.array([reye_x_min+head_loc[0],reye_y_min+head_loc[1],reye_x_max+head_loc[0],reye_y_max+head_loc[1]])


        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        # load the scene img
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        reserve_width,reserve_height=width,height
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if -1 in leye_loc:
            leye_img=np.zeros((36,60,3))
            leye_img=leye_img.astype(np.uint8)
            leye_img=Image.fromarray(leye_img)
        else:
            leye_img=img.crop((int(leye_loc[0]), int(leye_loc[1]), int(leye_loc[2]), int(leye_loc[3])))

        if -1 in reye_loc:
            reye_img=np.zeros((36,60,3))
            reye_img=reye_img.astype(np.uint8)
            reye_img=Image.fromarray(reye_img)
        else:
            reye_img=img.crop((int(reye_loc[0]), int(reye_loc[1]), int(reye_loc[2]), int(reye_loc[3])))

        # load the depth img
        depthimg=np.load(os.path.join(self.depth_dir,depth_path))
        depthimg=depthimg.astype(np.float32)


        if self.imshow:

            orgimg_show=img #img.resize((224,224))
            org_gazex,org_gazey=gaze_x,gaze_y
            org_eyex,org_eyey=eye_x,eye_y


        imsize = torch.IntTensor([width, height])
        if self.test:
            all_gaze_=cont_gaze[cont_gaze!=-1]
            all_gaze=all_gaze_.reshape(-1,2)
            gaze_mean=all_gaze.mean(0)

        ## data augmentation
        else:
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # maskimg=
                # depthimg=TF.crop(depthimg,crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                # else:
                #     gaze_x = -1; gaze_y = -1

                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)


                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

                leye_img=leye_img.transpose(Image.FLIP_LEFT_RIGHT)
                reye_img=reye_img.transpose(Image.FLIP_LEFT_RIGHT)

                leye_img,reye_img=reye_img,leye_img

                depthimg=np.fliplr(depthimg)

                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x =1 -eye_x


            # Random color change
            if np.random.random_sample() <= 0.5:

                b_f=np.random.uniform(0.5, 1.5)
                c_f=np.random.uniform(0.5, 1.5)
                s_f=np.random.uniform(0, 1.5)
                img = TF.adjust_brightness(img, brightness_factor=b_f)
                img = TF.adjust_contrast(img, contrast_factor=c_f)
                img = TF.adjust_saturation(img, saturation_factor=s_f)

                leye_img = TF.adjust_brightness(leye_img, brightness_factor=b_f)
                leye_img = TF.adjust_contrast(leye_img, contrast_factor=c_f)
                leye_img = TF.adjust_saturation(leye_img, saturation_factor=s_f)

                reye_img = TF.adjust_brightness(reye_img, brightness_factor=b_f)
                reye_img = TF.adjust_contrast(reye_img, contrast_factor=c_f)
                reye_img = TF.adjust_saturation(reye_img, saturation_factor=s_f)

        # head_channel = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
        #                                             resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # generate the gaze vector field
        gv_field = img_utils.generate_gaze_field(np.array([eye_x, eye_y]))

        # generate the front mid and back depthmap
        face_depth=depthimg[head_loc[1]:head_loc[3],head_loc[0]:head_loc[2]]

        Fd=depthimg-np.average(face_depth)



        Mfront=np.clip(Fd,0,1)
        Mmid=np.clip(1-self.tau*np.power(Fd,2),0,1)
        Mback=np.clip(-Fd,0,1)

        Mfront=Mfront[np.newaxis,:,:]
        Mmid=Mmid[np.newaxis,:,:]
        Mback=Mback[np.newaxis,:,:]

        mmap=np.concatenate([Mfront,Mmid,Mback],axis=0)
        mmap=mmap[np.newaxis,:,:,:]

        mmap=img_utils.to_torch(mmap)
        mmap=F.interpolate(mmap,self.depth_resize,mode="bilinear")
        mmap=mmap.squeeze(0)

        if (-1 in leye_loc) or (-1 in reye_loc):
            ind=False
        else:
            ind=True

        if self.imshow:
            # img_show=img
            face_show=face

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        if self.eye_transform is not None:
            l_eye=self.eye_transform(leye_img)
            r_eye=self.eye_transform(reye_img)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = img_utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = img_utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')

        # generate the gaze vector
        if self.test:

            gaze_vector=gaze_mean.numpy()-np.array([eye_x,eye_y])

        else:

            gaze_vector=np.array([gaze_x,gaze_y])-np.array([eye_x,eye_y])

        norm_gaze_vector = 1.0 if np.linalg.norm (gaze_vector) <= 0.0 else np.linalg.norm (gaze_vector)
        gaze_vector=gaze_vector/norm_gaze_vector

        gaze_vector=torch.from_numpy(gaze_vector)


        all_data={}
        if self.test:
            # X_train
            all_data['img']=img.float()
            all_data['mmimg']=mmap.float()
            all_data["face"]=face.float()
            all_data["l_eyeimg"]=l_eye.float()
            all_data["r_eyeimg"]=r_eye.float()
            all_data["indicator"]=ind
            all_data["gazefield"]=gv_field.float()

            # Y_label
            all_data["gaze_heatmap"]=gaze_heatmap.float()
            all_data["gaze_vector"]=gaze_vector
            all_data["gaze_inside"]=gaze_inside
            all_data["gaze_label"]=cont_gaze.float()

            all_data["imsize"]=imsize
        #
        #     return all_data
        #
        else:
            cont_gaze=np.array([gaze_x,gaze_y])
            cont_gaze=torch.FloatTensor(cont_gaze)

            # X_train
            all_data['img']=img.float()
            all_data['mmimg']=mmap.float()
            all_data["face"]=face.float()
            all_data["l_eyeimg"]=l_eye.float()
            all_data["r_eyeimg"]=r_eye.float()
            all_data["indicator"]=ind
            all_data["gazefield"]=gv_field.float()

            # Y_label
            all_data["gaze_heatmap"]=gaze_heatmap.float()
            all_data["gaze_vector"]=gaze_vector
            all_data["gaze_inside"]=gaze_inside
            all_data["gaze_label"]=cont_gaze.float()

            all_data["imsize"]=torch.IntTensor([0,0])

        return all_data

    def __len__(self):
        return self.length


def collate_fn(batch):

    batch_data={}

    batch_data["img"]=[]
    batch_data["mmimg"]=[]
    batch_data["face"]=[]
    batch_data["l_eyeimg"]=[]
    batch_data["r_eyeimg"]=[]
    batch_data["indicator"]=[]
    batch_data["gazefield"]=[]

    batch_data["gaze_heatmap"] = []
    batch_data["gaze_vector"] = []
    batch_data["gaze_inside"] = []
    batch_data["gaze_label"] = []

    batch_data["img_size"]=[]


    for data in batch:
        batch_data["img"].append(data["img"])
        batch_data["mmimg"].append(data["mmimg"])
        batch_data["face"].append(data["face"])
        batch_data["l_eyeimg"].append(data["l_eyeimg"])
        batch_data["r_eyeimg"].append(data["r_eyeimg"])
        batch_data["indicator"].append(data["indicator"])
        batch_data["gazefield"].append(data["gazefield"])

        batch_data["gaze_heatmap"].append(data["gaze_heatmap"])
        batch_data["gaze_vector"].append(data["gaze_vector"])
        batch_data["gaze_inside"].append(data["gaze_inside"])
        batch_data["gaze_label"].append(data["gaze_label"])

        batch_data["img_size"].append(data["imsize"])


    # train data
    batch_data["img"]=torch.stack(batch_data["img"],dim=0)
    batch_data["mmimg"]=torch.stack(batch_data["mmimg"],dim=0)
    batch_data["face"]=torch.stack(batch_data["face"],dim=0)
    batch_data["l_eyeimg"]=torch.stack(batch_data["l_eyeimg"],dim=0)
    batch_data["r_eyeimg"] = torch.stack(batch_data["r_eyeimg"], dim=0)
    batch_data["indicator"] = torch.as_tensor(batch_data["indicator"])
    batch_data["gazefield"] = torch.stack(batch_data["gazefield"], dim=0)


    # label data
    batch_data["gaze_heatmap"]=torch.stack(batch_data["gaze_heatmap"],0)
    batch_data["gaze_inside"] = torch.as_tensor(batch_data["gaze_inside"])
    batch_data["gaze_vector"] = torch.stack(batch_data["gaze_vector"], 0)
    batch_data["gaze_label"] = torch.stack(batch_data["gaze_label"], 0)

    batch_data["img_size"]=torch.stack(batch_data["img_size"],0)

    return batch_data
