import numpy as np
import pandas as pd
import cv2
import os
import math
from tqdm import tqdm
import glob


from preprocess.depth_estimation import DepthEstimator,read_image
from preprocess.eyes_extraction import EyesDetector

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

def concat_annotation(dataset="gazefollow",type="train"):

    org_train_path="../anno_files/{}/{}.txt".format(dataset,type)

    eye_coord_path="../anno_files/{}/eye_coord_{}.txt".format(dataset,type)

    org_train_pd=pd.read_csv(org_train_path)

    eye_train_pd=pd.read_csv(eye_coord_path)

    dam_train_pd=pd.concat([org_train_pd,eye_train_pd],axis=1)

    dam_train_pd.to_csv("../anno_files/{}/{}_{}_dam.txt".format(dataset,type,dataset),index=False)

    # pass

def extracteye_gazefollow(type="train"):

    eye_detector=EyesDetector()

    detect_eye_pd=pd.DataFrame(columns={"l_eye_xmin","l_eye_ymin","l_eye_xmax","l_eye_ymax",
                            "r_eye_xmin","r_eye_ymin","r_eye_xmax","r_eye_ymax"})

    root_dir="../datasets/gazefollow/"

    if type=="train":

        pd_path=os.path.join("../anno_files/gazefollow/train.txt")

        anno_pd = pd.read_csv(pd_path)

    elif type=="test":

        pd_path=os.path.join("../anno_files/gazefollow/test.txt")

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
            detect_eye_pd.to_csv("../anno_files/gazefollow/eye_coord_{}.txt".format(type), index=False)
        # print(head_loc,head_h,head_w)
        pbar.update(1)

    pbar.close()

    detect_eye_pd.to_csv("../anno_files/gazefollow/eye_coord_{}.txt".format(type),index=False)

def extracteye_videotargetatt(type="train"):

    eye_detector=EyesDetector()

    root_dir="../datasets/videoattentiontarget/images"

    # read the train.txt
    if type=="train":

        anno_pd=pd.read_csv("../anno_files/videoatt/train.txt")
    else:
        anno_pd=pd.read_csv("../anno_files/videoatt/test.txt")


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
            detect_eye_pd.to_csv("../anno_files/videoatt/eye_coord_{}.txt".format(type), index=False)

        pbar.update(1)
    pbar.close()

    detect_eye_pd.to_csv("../anno_files/videoatt/eye_coord_{}.txt".format(type), index=False)

    # head_img,R_matrix,t_p,map_p=rotate_img(head_img_org,135)


def depgen_videotargetatt():

    depthestimator=DepthEstimator()

    root_dir="./datasets/videoattentiontarget/images"
    save_root_dir="./datasets/videoattentiontarget/depdam"

    all_img_list=glob.glob(os.path.join(root_dir,"*","*","*"))

    pbar=tqdm(total=len(all_img_list))
    for img_path in all_img_list:

        all_path=img_path.split('/')

        save_dir=all_path[4:-1]
        save_path=all_path[4:]

        img=read_image(img_path)
        pred_depth = depthestimator.estimate(img)

        if pred_depth.max() > 0:
            pred_depth = pred_depth / pred_depth.max()
            pred_depth = np.clip(pred_depth, 0, 1)
        else:
            pred_depth = np.clip(pred_depth, 0, 1)
        # pred_depth_show=pred_depth.copy()

        pred_depth = pred_depth.astype(np.float16)


        depgen_dir=os.path.join(save_root_dir,"/".join(save_dir))

        if not os.path.exists(depgen_dir):
            os.makedirs(depgen_dir)

        depgen_path=os.path.join(save_root_dir,"/".join(save_path))

        np.save(depgen_path, pred_depth)

        pbar.update(1)

    pbar.close()


def depgen_gazefollow():

    depthestimator=DepthEstimator()

    # Crop the eye coordinate from the scene img
    gazefollow_root="./datasets/gazefollow"

    mode="train"

    # Generate the depthimg from the scene img
    if mode=="train":

        gazefollow_dir=os.path.join(gazefollow_root,"train")
        output_dir=os.path.join(gazefollow_root,"train_depth_dam")

    elif mode=="test":

        gazefollow_dir=os.path.join(gazefollow_root,"test")
        output_dir=os.path.join(gazefollow_root,"test_depth_dam")

    subdir_id_list=os.listdir(gazefollow_dir)

    subdir_id_list.sort()

    for curdir_id in subdir_id_list:

        print(curdir_id)
        curdir_add = os.path.join(gazefollow_dir, curdir_id)

        image_list = os.listdir(curdir_add)

        image_list.sort()

        output_add = os.path.join(output_dir, curdir_id)

        if not os.path.exists(output_add):

            os.makedirs(output_add)

        pbar = tqdm(total=len(image_list))
        for image in image_list:
            index, _ = image.split('.')

            img_path = os.path.join(gazefollow_dir, curdir_id, image)

            img = read_image(img_path)

            pred_depth=depthestimator.estimate(img)

            if pred_depth.max()>0:
                pred_depth=pred_depth/pred_depth.max()
                pred_depth=np.clip(pred_depth,0,1)
            else:
                pred_depth=np.clip(pred_depth,0,1)
            # pred_depth_show=pred_depth.copy()

            pred_depth=pred_depth.astype(np.float16)

            np.save(os.path.join(output_add, index), pred_depth)

            pbar.update(1)

        pbar.close()




if __name__ == '__main__':

    # Crop the eye coordinate from the scene img

    # On gazefollow dataset
    extracteye_gazefollow("train")
    extracteye_gazefollow("test")

    concat_annotation("gazefollow",type="train")
    concat_annotation("gazefollow",type="test")


    # On videoattentiontarget
    extracteye_videotargetatt("train")
    extracteye_videotargetatt("test")
    #
    concat_annotation("videoatt",type="train")
    concat_annotation("videoatt",type="test")


    # Generate the depth image from the scene img
    # On gazefollow dataset
    depgen_gazefollow()

    # On videoattentiontarget
    depgen_videotargetatt()

