import torch
from torch.utils.data import Dataset
import scipy.io as scio
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

gaze360mat_path="/home/nku120/HZX/dataset/Gaze360/metadata.mat"
root_path="/home/nku120/HZX/dataset/Gaze360/imgs"

cropType='head'

if __name__ == '__main__':


    gaze360mat=scio.loadmat(gaze360mat_path)


    recording=gaze360mat['recording']
    recording=recording.squeeze()

    # recordings
    recordings=gaze360mat['recordings']
    recordings=recordings.squeeze()
    recordings=[i[0] for i in recordings]

    # person id
    person_identity=gaze360mat['person_identity']
    person_identity=person_identity.squeeze()

    # frame
    frame=gaze360mat['frame']
    frame=frame.squeeze()

    # label
    splits=gaze360mat['splits']
    splits=splits.squeeze()
    splits=[i[0] for i in splits]

    split=gaze360mat['split']
    split=split.squeeze()

    gaze_dir_in_eye_coord=gaze360mat['gaze_dir']

    person_head_bbox=gaze360mat['person_head_bbox']
    person_eye_left_bbox=gaze360mat['person_eye_left_bbox']
    person_eye_right_bbox=gaze360mat['person_eye_right_bbox']


    rotate_M=np.eye(3,3)
    # rotate_M[0,0]=-1
    # rotate_M[1,1]=-1


    # rotate the gaze direction to image coordinate
    gaze_dir_in_image_coord=np.dot(rotate_M,gaze_dir_in_eye_coord.T)
    gaze_dir_in_image_coord=gaze_dir_in_image_coord.T

    test_pd=pd.DataFrame(columns=['path','gv_x','gv_y','gv_z',
                                  'lbbox_x','lbbox_y','lbbox_w','lbbox_h',
                                  'rbbox_x','rbbox_y','rbbox_w','rbbox_h'])
    train_pd=pd.DataFrame(columns=['path','gv_x','gv_y','gv_z',
                                   'lbbox_x', 'lbbox_y', 'lbbox_w', 'lbbox_h',
                                   'rbbox_x', 'rbbox_y', 'rbbox_w', 'rbbox_h'])
    valid_pd=pd.DataFrame(columns=['path','gv_x','gv_y','gv_z',
                                   'lbbox_x', 'lbbox_y', 'lbbox_w', 'lbbox_h',
                                   'rbbox_x', 'rbbox_y', 'rbbox_w', 'rbbox_h'])


    for index in tqdm(range(len(recording))):

        cur_records=recordings[recording[index]]

        cur_person_id=person_identity[index]

        cur_frame=frame[index]

        cur_img_path=os.path.join(root_path,cur_records,cropType,'%06d'%cur_person_id,'%06d.jpg'%cur_frame)

        absolute_path=cur_img_path.replace(root_path,"")[1:]

        cur_split=splits[split[index]]

        cur_gaze_dir=gaze_dir_in_image_coord[index]

        cur_head_bbox=person_head_bbox[index]
        cur_eye_rbbox=person_eye_right_bbox[index]
        cur_eye_lbbox=person_eye_left_bbox[index]

        # print(cur_eye_lbbox,cur_eye_rbbox)

        if -1 in cur_eye_rbbox:
            cur_eye_right_crop=[-1,-1,-1,-1]

        else:

            cur_eye_right_crop=[(cur_eye_rbbox[0]-cur_head_bbox[0])/cur_head_bbox[2],
                          (cur_eye_rbbox[1]-cur_head_bbox[1])/cur_head_bbox[3],
                          cur_eye_rbbox[2]/cur_head_bbox[2],
                          cur_eye_rbbox[3]/cur_head_bbox[3]]

        if -1 in cur_eye_lbbox:
            cur_eye_left_crop=[-1,-1,-1,-1]
        else:

            cur_eye_left_crop = [(cur_eye_lbbox[0] - cur_head_bbox[0]) / cur_head_bbox[2],
                                  (cur_eye_lbbox[1] - cur_head_bbox[1]) / cur_head_bbox[3],
                                  cur_eye_lbbox[2] / cur_head_bbox[2],
                                  cur_eye_lbbox[3] / cur_head_bbox[3]]

        pd.set_option('precision',16)
        data_info=pd.Series({'path':absolute_path,'gv_x':float(cur_gaze_dir[0]),'gv_y':cur_gaze_dir[1],'gv_z':cur_gaze_dir[2],
                             'lbbox_x':cur_eye_left_crop[0],'lbbox_y':cur_eye_left_crop[1],'lbbox_w':cur_eye_left_crop[2],'lbbox_h':cur_eye_left_crop[3],
                             'rbbox_x': cur_eye_right_crop[0], 'rbbox_y': cur_eye_right_crop[1],'rbbox_w': cur_eye_right_crop[2], 'rbbox_h': cur_eye_right_crop[3],
                             })



        if cur_split=="train":
            train_pd=train_pd.append(data_info,ignore_index=True)
        elif cur_split=="val":

            valid_pd=valid_pd.append(data_info,ignore_index=True)

        elif cur_split=="test":

            test_pd=test_pd.append(data_info,ignore_index=True)
        elif cur_split=="unused":
            pass
        else:
            raise NotImplemented


    train_pd.to_csv('train_eye.txt',header=None,index=False,float_format='%.15f')

    valid_pd.to_csv('val_eye.txt',header=None,index=False,float_format='%.15f')

    test_pd.to_csv('test_eye.txt',header=None,index=False,float_format='%.15f')








