import pandas as pd
import os
import glob
import numpy as np
from preprocess.utils import getRootPath

def concat_annotation(dataset="gazefollow",type="train"):
    proj_path = getRootPath()

    if dataset=="gazefollow":

        if type=="train":


            org_train_pd=pd.read_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_train_anno.txt"))


            eye_train_pd=pd.read_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/eye_coord_train.txt"))
            eye_train_pd=eye_train_pd.iloc[:,1:]

            dam_train_pd=pd.concat([org_train_pd,eye_train_pd],axis=1)

            dam_train_pd.to_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/train_annotation_dam.txt"),index=False)

        elif type=="test":

            org_test_pd=pd.read_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_test_anno.txt"))

            value_counts=org_test_pd['rgb_path'].value_counts()

            eye_test_pd=pd.read_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/eye_coord_test.txt"))

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

            dam_test_pd.to_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/test_annotation_dam.txt"), index=False)

        else:
            raise NotImplemented

    elif dataset=="videoatt":

        org_train_path=os.path.join(proj_path,"datasets/videotargetattention_annotations/vat_{}_anno.txt".format(type))

        eye_coord_path=os.path.join(proj_path,"datasets/videotargetattention_annotations/eye_coord_{}.txt".format(type))

        save_path=os.path.join(proj_path,"datasets/videotargetattention_annotations/{}_annotation_dam.txt".format(type))

        org_train_pd=pd.read_csv(org_train_path)

        eye_train_pd=pd.read_csv(eye_coord_path)

        dam_train_pd=pd.concat([org_train_pd,eye_train_pd],axis=1)

        dam_train_pd.to_csv(save_path,index=False)

    else:
        raise NotImplemented


def integrate_annofile(dataset="gazefollow",type="train"):

    proj_path=getRootPath()

    if dataset=="gazefollow":

        save_dir=os.path.join(proj_path,"datasets/gazefollow_annotations/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if type=="train":


            train_pd = pd.read_csv(os.path.join(proj_path,"datasets/gazefollow/annotations_extend/train_annotations_release.txt"),header=None)
            train_pd.columns=["rgb_path","id","body_bbox_x","body_bbox_y","body_bbox_width","body_bbox_height","eye_x","eye_y","gaze_x","gaze_y",
                              "head_bbox_x_min","head_bbox_y_min","head_bbox_x_max","head_bbox_y_max","in_or_out","meta","meta2"]
            depth_path=train_pd["rgb_path"]
            depth_path=depth_path.str.replace('train','train_depth')
            depth_path = depth_path.str.replace('.jpg', '.npy')
            train_pd.insert(loc=1,column="depth_path",value=depth_path)
            train_pd.to_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_train_anno.txt"),index=False)

        elif type=="test":
            test_pd= pd.read_csv(os.path.join(proj_path,"datasets/gazefollow/annotations_extend/test_annotations_release.txt"),header=None)
            test_pd.columns=["rgb_path","id","body_bbox_x","body_bbox_y","body_bbox_width","body_bbox_height","eye_x","eye_y","gaze_x","gaze_y",
                              "head_bbox_x_min","head_bbox_y_min","head_bbox_x_max","head_bbox_y_max","meta","meta2"]
            test_pd['rgb_path']=test_pd['rgb_path'].str.replace('test2','test')

            depth_path=test_pd["rgb_path"]
            depth_path=depth_path.str.replace('test','test_depth')
            depth_path = depth_path.str.replace('.jpg', '.npy')
            test_pd.insert(loc=1,column="depth_path",value=depth_path)


            test_pd.to_csv(os.path.join(proj_path,"datasets/gazefollow_annotations/gf_test_anno.txt"),index=False)


    elif dataset=="videoatt":

        save_dir=os.path.join(proj_path,"datasets/videoattentiontarget/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        video_list = os.listdir(os.path.join(proj_path,"datasets/videoattentiontarget/", 'images'))
        video_list.sort()

        anno_dir=os.path.join(proj_path,"datasets/videoattentiontarget/annotations")

        save_dir=os.path.join(proj_path,"datasets/videotargetattention_annotations")

        # num2str = {i: (
        # each, 'vat_train' if each in os.listdir(os.path.join(self.anno_dir, 'train')) else 'vat_test') for
        #            i, each in enumerate(video_list)}
        str2num = {each: (
        i, 'vat_train' if each in os.listdir(os.path.join(anno_dir, 'train')) else 'vat_test') for
                   i, each in enumerate(video_list)}

        df = pd.DataFrame(
            columns=['sub_id', 'act_id', 'show_index', 'show_name', 'frame_scope', 'img_name', 'x_min', 'y_min',
                     'x_max', 'y_max',
                     'gaze_x', 'gaze_y'])

        shows = glob.glob(os.path.join(anno_dir, type, '*', '*', '*.txt'))
        for s in shows:

            s_string = s.replace(os.path.join(anno_dir, type), "")
            cur_pd = pd.read_csv(s, header=None, index_col=False,
                                 names=['img_name', 'x_min', 'y_min', 'x_max', 'y_max', 'gaze_x', 'gaze_y'])

            _, show_name, frame_scope, anno_file = s_string.split('/')

            # used to associate the subject name in annotation  and actual subject ids in features
            cur_anno_dir = os.path.join(anno_dir, type, show_name, frame_scope)
            cur_anno_files_list = os.listdir(cur_anno_dir)
            cur_anno_files_list = sorted(cur_anno_files_list, key=lambda x: (x[1:3]))
            map_dict = {}
            counter = 0
            for cur_anno_file in cur_anno_files_list:
                cur_anno_file_name = cur_anno_file[0:3]
                map_dict[cur_anno_file_name] = counter
                counter += 1

            cur_pd['show_name'] = show_name
            cur_pd['frame_scope'] = frame_scope
            cur_pd['sub_id'] = anno_file[0:3]
            cur_pd['act_id'] = map_dict[anno_file[0:3]]
            cur_pd['show_index'] = str2num[show_name][0]

            df = pd.concat([df, cur_pd], axis=0, join='inner', ignore_index=True)

        df.to_csv(os.path.join(save_dir, 'vat_'+type+'_anno' + '.txt'), index=False)



