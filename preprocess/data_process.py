import os
import glob
import numpy as np
from tqdm import tqdm

from preprocess.depth_estimation import DepthEstimator,read_image
import matplotlib.pyplot as plt
# from preprocess.eyes_extraction import EyesDetector

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

if __name__ == '__main__':

    # Crop the eye coordinate from the scene img

    # Generate the depthimg from the scene img
    depgen_videotargetatt()

    pass