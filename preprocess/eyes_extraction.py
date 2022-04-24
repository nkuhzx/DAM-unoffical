import face_alignment
import numpy as np
import math
import collections
import pandas as pd
import os
from tqdm import tqdm
import cv2

class EyesDetector(object):

    def __init__(self):
        super(EyesDetector,self).__init__()

        face_detector_kwargs = {
            "filter_threshold": 0.8
        }

        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True,
                                                     face_detector='sfd', face_detector_kwargs=face_detector_kwargs)

    def first_check(self,headimg):

        pass

    # Bulat et al 's method
    def second_check(self,headimg):
        try:

            preds=self.face_detector.get_landmarks(headimg)[-1]

            pred_type=collections.namedtuple('prediction_type',['slice','color'])
            pred_types={
                "eye1":pred_type(slice(36,42),(0.596,0.875,0.541,0.3)),
                "eye2":pred_type(slice(42,48),(0.596,0.875,0.541,0.3))
            }

            centers=[]

            Avg_eye=24 #24 represent average eye length for adults, we use this to set the scale

            for pred_type in pred_types.values():
                x=preds[pred_type.slice,0]
                y=preds[pred_type.slice,1]

                centroid=(sum(x)/len(x),sum(y)/len(y))
                centers.append(centroid)


            dist=math.hypot(centers[0][0]-centers[1][0],centers[0][1]-centers[1][1])
            dist_real=np.divide(dist*Avg_eye,x.max()-x.min())

            if (51<dist_real<77):

                centers=[(int(element[0]),int(element[1])) for element in centers]

                r_eye_crop=[centers[0][0]-7,centers[0][0]+7,centers[0][1]-7,centers[0][1]+7]
                l_eye_crop=[centers[1][0]-7,centers[1][0]+7,centers[1][1]-7,centers[1][1]+7]

            else:

                r_eye_crop=[-1,-1,-1,-1]
                l_eye_crop=[-1,-1,-1,-1]

        except:

            r_eye_crop=[-1,-1,-1,-1]
            l_eye_crop = [-1, -1, -1, -1]

        # for show
        self.headimg=headimg
        self.l_eye_coord=l_eye_crop
        self.r_eye_coord=r_eye_crop

        return l_eye_crop,r_eye_crop


    def visualization(self):

        cv2.rectangle(self.headimg, (self.r_eye_coord[0], self.r_eye_coord[2]), (self.r_eye_coord[1], self.r_eye_coord[3]), color=(0, 255, 255),
                      thickness=1, lineType=4)
        cv2.rectangle(self.headimg, (self.l_eye_coord[0], self.l_eye_coord[2]), (self.l_eye_coord[1], self.l_eye_coord[3]),
                      color=(0, 0, 255), thickness=1, lineType=4)
        cv2.imshow("head", self.headimg)
        cv2.waitKey(0)




    def all_process(self,headimg):

        # self.first_check(headimg)

        self.second_check(headimg)




def train_eye_coord():

    eye_detector=EyesDetector()

    detect_eye_pd=pd.DataFrame(columns={"rgb_path","l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"})

    root_dir="/home/nku120/HZX/dataset/gazefollow"

    train_pd_path=os.path.join(root_dir,"train_annotations_revised.txt")

    train_pd=pd.read_csv(train_pd_path)

    pbar=tqdm(total=len(train_pd))

    for index ,row in train_pd.iterrows():

        head_loc=row["head_bbox_x_min":"head_bbox_y_max"]
        head_loc=np.array(head_loc).astype(int)

        org_rgb_path=row["rgb_path"]
        # load the information
        rgb_path=os.path.join(root_dir,row["rgb_path"])

        head_w=head_loc[2]-head_loc[0]
        head_h=head_loc[3]-head_loc[1]

        rgb_img=cv2.imread(rgb_path)

        head_img=rgb_img[head_loc[1]:head_loc[3],head_loc[0]:head_loc[2]]

        l_eye,r_eye=eye_detector.second_check(head_img)
        # cv2.imshow("head_img",head_img)
        # cv2.waitKey(100)

        s=pd.Series({
            'rgb_path':str(org_rgb_path),
            "l_eye_xmin":l_eye[0] ,
            "l_eye_ymin":l_eye[2],
            "l_eye_xmax":l_eye[1],
            "l_eye_ymax":l_eye[3],
            "r_eye_xmin":r_eye[0],
            "r_eye_ymin":r_eye[2],
            "r_eye_xmax":r_eye[1],
            "r_eye_ymax":r_eye[3]
        })

        detect_eye_pd= detect_eye_pd.append(s,ignore_index=True)

        detect_eye_pd=detect_eye_pd[["rgb_path","l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"]]

        if index%10000==0 and index!=0:
            detect_eye_pd.to_csv("eye_coord_train.txt", index=False)
        # print(head_loc,head_h,head_w)
        pbar.update(1)

    pbar.close()

    detect_eye_pd.to_csv("eye_coord_train.txt",index=False)


def test_eye_coord():

    eye_detector=EyesDetector()

    detect_eye_pd=pd.DataFrame(columns={"rgb_path","l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"})

    root_dir="/home/nku120/HZX/dataset/gazefollow"

    test_pd_path=os.path.join(root_dir,"test_annotations_revised.txt")

    test_pd=pd.read_csv(test_pd_path)

    test_pd.drop_duplicates(subset=['rgb_path'],inplace=True,keep='first')

    pbar=tqdm(total=len(test_pd))

    for index ,row in test_pd.iterrows():

        head_loc=row["head_bbox_x_min":"head_bbox_y_max"]
        head_loc=np.array(head_loc).astype(int)

        org_rgb_path=row["rgb_path"]
        # load the information
        rgb_path=os.path.join(root_dir,row["rgb_path"])

        head_w=head_loc[2]-head_loc[0]
        head_h=head_loc[3]-head_loc[1]

        rgb_img=cv2.imread(rgb_path)

        head_img=rgb_img[head_loc[1]:head_loc[3],head_loc[0]:head_loc[2]]

        l_eye,r_eye=eye_detector.second_check(head_img)
        # cv2.imshow("head_img",head_img)
        # cv2.waitKey(100)

        s=pd.Series({
            'rgb_path':str(org_rgb_path),
            "l_eye_xmin":l_eye[0] ,
            "l_eye_ymin":l_eye[2],
            "l_eye_xmax":l_eye[1],
            "l_eye_ymax":l_eye[3],
            "r_eye_xmin":r_eye[0],
            "r_eye_ymin":r_eye[2],
            "r_eye_xmax":r_eye[1],
            "r_eye_ymax":r_eye[3]
        })

        detect_eye_pd= detect_eye_pd.append(s,ignore_index=True)

        detect_eye_pd=detect_eye_pd[["rgb_path","l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"]]

        if index%1000==0 and index!=0:
            detect_eye_pd.to_csv("eye_coord_test.txt", index=False)
        # print(head_loc,head_h,head_w)
        pbar.update(1)

    pbar.close()

    detect_eye_pd.to_csv("eye_coord_test.txt",index=False)


def concat_annotation(type="train"):

    if type=="train":

        org_train_pd=pd.read_csv("/home/nku120/HZX/dataset_process/DAM_process/train_annotations_revised.txt")


        eye_train_pd=pd.read_csv("/home/nku120/HZX/dataset_process/DAM_process/preprocess/eye_coord_train.txt")
        eye_train_pd=eye_train_pd.iloc[:,1:]

        dam_train_pd=pd.concat([org_train_pd,eye_train_pd],axis=1)


        dam_train_pd.to_csv("./train_annotation_dam.txt",index=False)

    elif type=="test":

        org_test_pd=pd.read_csv("/home/nku120/HZX/dataset_process/DAM_process/test_annotations_revised.txt")

        value_counts=org_test_pd['rgb_path'].value_counts()

        eye_test_pd=pd.read_csv("/home/nku120/HZX/dataset_process/DAM_process/preprocess/eye_coord_test.txt")



        all_df=[]
        for i in range(len(eye_test_pd)):

            rgb_path=eye_test_pd.iloc[i,0]

            cur_series=eye_test_pd.iloc[i:i+1,:]
            cur_sumvalue=value_counts[rgb_path]

            temp_df=pd.DataFrame(np.repeat(cur_series.values,cur_sumvalue,axis=0))
            temp_df.columns=cur_series.columns
            all_df.append(temp_df)

        eye_test_pd=pd.concat(all_df,axis=0)

        eye_test_pd.reset_index(inplace=True,drop=True)
        eye_test_pd=eye_test_pd.iloc[:,1:]

        dam_test_pd = pd.concat([org_test_pd, eye_test_pd], axis=1)

        dam_test_pd.to_csv("./test_annotation_dam.txt", index=False)




    else:
        raise NotImplemented


if __name__ == '__main__':
    # test_eye_coord()
    # concat_annotation(type="test")

    # test_pd=pd.read_csv("/home/nku120/HZX/dataset_process/DAM_process/preprocess/test_annotation_dam.txt")
    #
    # test_pd.rename(columns={'masks_path':'depth_path'},inplace=True)
    #
    # split_pd=test_pd["depth_path"].str.split('/',expand=True)
    #
    # split_pd.iloc[:,0]="test_depth"
    #
    # split_pd["sum"]=split_pd.iloc[:,0]+"/"+split_pd.iloc[:,1]+"/"+split_pd.iloc[:,2]
    # test_pd["depth_path"]=split_pd["sum"]
    #
    # test_pd.to_csv("./test_annotation_dam.txt", index=False)
    eye_detector=EyesDetector()
    #
    # head_img=cv2.imread("/home/nku120/HZX/dataset/Gaze360/imgs/rec_000/head/000068/002098.jpg")
    #
    # eye_detector.second_check(head_img)
    #
    # eye_detector.visualization()





