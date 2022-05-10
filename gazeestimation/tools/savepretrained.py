import os
import torch

def getRootPath():

    rootPath=os.path.dirname(os.path.abspath(__file__))

    rootPath=os.path.dirname(rootPath)
    rootPath=os.path.dirname(rootPath)

    return rootPath

if __name__ == '__main__':

    proj_path=getRootPath()

    checkpoints_dir=os.path.join(proj_path,"gazeestimation/checkpoints")

    save_path=os.path.join(proj_path,"dammethod/modelparas")

    checkpoints_list=os.listdir(checkpoints_dir)

    checkpoints_list=sorted(checkpoints_list,key=lambda x:x[8])

    checkpoint=checkpoints_list[-1]

    pretrained_weight=torch.load(os.path.join(checkpoints_dir,checkpoint))["state_dict"]

    torch.save(pretrained_weight,os.path.join(save_path,"pretrained.pt"))

