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

