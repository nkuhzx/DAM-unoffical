import numpy as np
import torch
from collections import deque

def Angularto3d(angular):

    gaze_v=torch.zeros([3])
    gaze_v[0]=-torch.cos(angular[1])*torch.sin(-angular[0])
    gaze_v[1]=torch.sin(angular[1])
    gaze_v[2]=-torch.cos(angular[1])*torch.cos(-angular[0])

    return gaze_v

def Angularto3d_matrix(angular):

    gaze_v=torch.zeros([angular.shape[0],3])
    gaze_v[:,0]=-torch.cos(angular[:,1])*torch.sin(-angular[:,0])
    gaze_v[:,1]=torch.sin(angular[:,1])
    gaze_v[:,2]=-torch.cos(angular[:,1])*torch.cos(-angular[:,0])

    return gaze_v


class AverageMeter():

    def __init__(self):

        self.reset()

    def reset(self):

        self.count=0
        self.newval=0
        self.sum=0
        self.avg=0

    def update(self,newval,n=1):

        self.newval=newval
        self.sum+=newval*n
        self.count+=n
        self.avg=self.sum/self.count

class MovingAverageMeter():

    def __init__(self,max_len=30):

        self.max_len=max_len

        self.reset()

    def reset(self):

        self.dq=deque(maxlen=self.max_len)
        self.count=0
        self.avg=0
        self.sum=0


    def update(self,newval):

        self.dq.append(newval)
        self.count=len(self.dq)
        self.sum=np.array(self.dq).sum()
        self.avg=self.sum/float(self.count)


