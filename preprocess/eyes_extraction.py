import face_alignment
import numpy as np
import math
import collections
import pandas as pd
import os
from tqdm import tqdm
import cv2
from preprocess.utils import getRootPath

class EyesDetector(object):

    def __init__(self):
        super(EyesDetector,self).__init__()

        face_detector_kwargs = {
            "filter_threshold": 0.8
        }

        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True,
                                                     face_detector='sfd', face_detector_kwargs=face_detector_kwargs)

        self.rotate_ang_list=[45,90,270,315]



    def first_check(self,headimg):

        pass

    def rotate_img(self,headimg,angle):

        h,w=headimg.shape[:2]
        cx,cy=w/2,h/2

        #
        M=cv2.getRotationMatrix2D((cx,cy),angle,1.0)
        cos=np.abs(M[0,0])
        sin=np.abs(M[0,1])

        # compute the new bounding dimensions of the image
        nW = int((h*sin)+(w*cos))
        nH = int((h*cos)+(w*sin))

        # adjust the rotation matrix to take into account translation
        M[0,2]+=(nW/2)-cx
        M[1,2]+=(nH/2)-cy

        nheadimg=cv2.warpAffine(headimg,M,(nW,nH))

        return nheadimg,M

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

        org_H,org_W=headimg.shape[:2]
        # self.first_check(headimg)

        # normal detect
        l_eye_coord,r_eye_coord=self.second_check(headimg)

        # normal detect false then rotate to detect
        if -1 in l_eye_coord or -1 in r_eye_coord:

            l_eye_coord_list = []
            r_eye_coord_list = []
            # rotate the img for detect
            for rotate_ang in self.rotate_ang_list:

                # obtain the rotate headimg and rotate matrix (2x3)
                r_headimg,r_matrix=self.rotate_img(headimg,rotate_ang)

                n_H,n_W=r_headimg.shape[:2]

                # get the inverse rotation matrix (3x3)
                R = r_matrix[0:2, 0:2]
                bias = r_matrix[:, 2]
                biasT = np.matmul(-R.T, bias)
                r_matrixT = np.zeros((3, 3))
                r_matrixT[2, 2] = 1
                r_matrixT[0:2, 0:2] = R.T
                r_matrixT[:2, 2] = biasT.T

                # detect the rotate image
                l_eye_coord,r_eye_coord= self.second_check(r_headimg)

                if -1 not in l_eye_coord and -1 not in r_eye_coord:

                    l_eye_coord=np.array(l_eye_coord).reshape(2,2)

                    r_eye_coord=np.array(r_eye_coord).reshape(2,2)

                    all_eye_coord=np.concatenate([l_eye_coord,r_eye_coord],axis=1)

                    all_eye_coord=np.pad(all_eye_coord,((0,1),(0,0)),'constant',constant_values=(0,0))
                    all_eye_coord=np.matmul(r_matrixT,all_eye_coord-np.repeat(np.array([[n_W/2.],[n_H/2.],[0]]),4,axis=1))

                    all_eye_coord=all_eye_coord[:2,:]+np.repeat(np.array([[org_W/2.],[org_H/2.]]),4,axis=1)

                    l_center_coord=np.average(all_eye_coord[:,0:2],axis=1)
                    r_center_coord=np.average(all_eye_coord[:,2:],axis=1)

                    r_eye_coord = np.array([r_center_coord[0] - 7, r_center_coord[0] + 7, r_center_coord[1] - 7, r_center_coord[1] + 7])
                    l_eye_coord = np.array([l_center_coord[0] - 7, l_center_coord[0] + 7, l_center_coord[1] - 7, l_center_coord[1] + 7])

                    # l_eye_coord=all_eye_coord[:,0:2].reshape(-1,)
                    # r_eye_coord=all_eye_coord[:,2:].reshape(-1,)

                    l_eye_coord=l_eye_coord.astype(int)
                    r_eye_coord=r_eye_coord.astype(int)



                l_eye_coord_list.append(l_eye_coord)
                r_eye_coord_list.append(r_eye_coord)


            l_eye_coord_list=np.concatenate(l_eye_coord_list,axis=0).reshape(-1,4)
            r_eye_coord_list=np.concatenate(r_eye_coord_list,axis=0).reshape(-1,4)

            l_choice_index=np.sum(l_eye_coord_list!=-1,axis=1).astype(bool)
            r_choice_index = np.sum(r_eye_coord_list != -1, axis=1).astype(bool)

            l_eye_coord_list=l_eye_coord_list[l_choice_index]
            r_eye_coord_list=r_eye_coord_list[r_choice_index]

            if l_eye_coord_list.shape[0]==0 :
                l_eye_coord=[-1,-1,-1,-1]
            else:
                l_eye_coord=np.average(l_eye_coord_list,axis=0)
                l_eye_coord = list(l_eye_coord.astype(int))

            if r_eye_coord_list.shape[0]==0:
                r_eye_coord=[-1,-1,-1,-1]
            else:
                r_eye_coord=np.average(r_eye_coord_list,axis=0)
                r_eye_coord=list(r_eye_coord.astype(int))

        return l_eye_coord,r_eye_coord


def extracteye_gazefollow(type="train"):

    proj_path=getRootPath()

    eye_detector=EyesDetector()

    detect_eye_pd=pd.DataFrame(columns={"l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"})

    root_dir=os.path.join(proj_path,"datasets/gazefollow/")

    save_path = os.path.join(proj_path, "datasets/gazefollow_annotations/eye_coord_{}.txt".format(type))

    if type=="train":

        pd_path=os.path.join(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_train_anno.txt"))

        anno_pd = pd.read_csv(pd_path)

    elif type=="test":

        pd_path=os.path.join(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_test_anno.txt"))

        anno_pd = pd.read_csv(pd_path)

        anno_pd.drop_duplicates(subset=['rgb_path'], inplace=True, keep='first')

    pbar=tqdm(total=len(anno_pd))

    for index ,row in anno_pd.iterrows():

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

        detect_eye_pd=detect_eye_pd[["l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"]]

        if index%10000==0 and index!=0:

            detect_eye_pd.to_csv(save_path, index=False)
        # print(head_loc,head_h,head_w)
        pbar.update(1)

    pbar.close()

    detect_eye_pd.to_csv(save_path,index=False)

def extracteye_videotargetatt(type="train"):

    proj_path=getRootPath()

    eye_detector=EyesDetector()

    root_dir=os.path.join(proj_path,"datasets/videoattentiontarget/images")

    save_path=os.path.join(proj_path, "datasets/videotargetattention_annotations/eye_coord_{}.txt".format(type))
    # read the train.txt
    if type=="train":

        anno_pd=pd.read_csv(os.path.join(proj_path,"datasets/videotargetattention_annotations/vat_train_anno.txt"))
    else:
        anno_pd=pd.read_csv(os.path.join(proj_path,"datasets/videotargetattention_annotations/vat_test_anno.txt"))


    detect_eye_pd=pd.DataFrame(columns={"l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"})

    pbar=tqdm(total=len(anno_pd))

    for index, row in anno_pd.iterrows():

        head_loc=row["x_min":"y_max"]
        head_loc=np.array(head_loc).astype(int)

        rgb_path=os.path.join(root_dir,row["show_name"],row["frame_scope"],row["img_name"])

        rgb_img=cv2.imread(rgb_path)

        head_img_org=rgb_img[head_loc[1]:head_loc[3],head_loc[0]:head_loc[2]]

        l_eye,r_eye=eye_detector.all_process(head_img_org)

        s=pd.Series({
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

        detect_eye_pd=detect_eye_pd[["l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"]]

        if index%10000==0 and index!=0:
            detect_eye_pd.to_csv(save_path.format(type), index=False)

        pbar.update(1)
    pbar.close()

    detect_eye_pd.to_csv(save_path.format(type), index=False)





