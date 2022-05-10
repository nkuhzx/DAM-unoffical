from yacs.config import CfgNode as CN
import os

def getRootPath():

    rootPath=os.path.dirname(os.path.abspath(__file__))


    rootPath=rootPath.split("/dammethod/config")[0]

    return rootPath
# -----------------------------------------------------------------------------
# Default Config definition
# -----------------------------------------------------------------------------

_C=CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET=CN()

_C.DATASET.root_dir = os.path.join(getRootPath(),"datasets/gazefollow")
_C.DATASET.train_anno = os.path.join(getRootPath(),"datasets/gazefollow/train_gazefollow_dam.txt")
_C.DATASET.test_anno = os.path.join(getRootPath(),"datasets/gazefollow/test_gazefollow_dam.txt")


# dataset loader
_C.DATASET.load_workers=24
_C.DATASET.train_batch_size=64
_C.DATASET.test_batch_size=64

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL=CN()

_C.MODEL.inout_branch=False


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

_C.TRAIN=CN()

# pre-trained parameters
_C.TRAIN.gazeestipretrain=os.path.join(getRootPath(),"dammethod/modelparas/gazedirmodel.pt")

_C.TRAIN.criterion="mixed"
_C.TRAIN.optimizer="adam"

_C.TRAIN.maxlr=1e-4
_C.TRAIN.minlr=0.000001
_C.TRAIN.weightDecay=1e-4
_C.TRAIN.epsilion=1e-8

_C.TRAIN.start_epoch=0
_C.TRAIN.end_epoch=40

_C.TRAIN.stage=1

# input and output resolution
_C.TRAIN.input_size=224
_C.TRAIN.output_size=64

_C.TRAIN.eye_size=[36,60]

# model parameters save interval and address
_C.TRAIN.store=os.path.join(getRootPath(),"modelparas/savemodel")
_C.TRAIN.save_intervel=1

# model parameters resume and initmodel
_C.TRAIN.resume=False
_C.TRAIN.resume_add=''

_C.TRAIN.initmodel=False
_C.TRAIN.initmodel_add=''


# -----------------------------------------------------------------------------
# Other Default
# -----------------------------------------------------------------------------
_C.OTHER=CN()


_C.OTHER.seed=235
# if gpu is used
_C.OTHER.device='cpu'

# log for tensorboardx
_C.OTHER.logdir='../logs'

_C.OTHER.global_step=0

_C.OTHER.lossrec_every=10

_C.OTHER.evalrec_every=600




