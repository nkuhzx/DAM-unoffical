import torch
import torch.nn.functional as F
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class DepthEstimator():

    def __init__(self,device="cuda"):

        model_type="DPT_Large"
        midas=torch.hub.load("intel-isl/MiDaS",model_type)
        midas_transforms=torch.hub.load("intel-isl/MiDaS","transforms")

        self.device=torch.device(device)
        midas.to(device)


        self.midas=midas

        if model_type=="DPT_Large" or model_type=="DPT_Hybrid":

            self.transform=midas_transforms.dpt_transform
        else:
            self.transform=midas_transforms.small_transform


    @torch.no_grad()
    def estimate(self,imgs):

        input_img=self.transform(imgs).to(self.device)

        pred_depth=self.midas(input_img)

        pred_depth=F.interpolate(
            pred_depth.unsqueeze(1),
            size=imgs.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        output=pred_depth.cpu().numpy()

        return output

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #/ 255.0

    return img


if __name__ == '__main__':

    depthestimator=DepthEstimator()

    # Crop the eye coordinate from the scene img
    gazefollow_root="/home/hzx/project/DAM_process/datasets/gazefollow"

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
            # plt.imshow(pred_depth_show,cmap='jet')

            # plt.show()