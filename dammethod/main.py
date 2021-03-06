import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
import torch
import torch.backends.cudnn as cudnn
import argparse
import shutil
import random
import time
import numpy as np
from datetime import datetime

from config import cfg
from dammethod.dataset.gazefollow import GazeFollowLoader

from dammethod.utils.model_utils import init_model,setup_model,save_checkpoint,resume_checkpoint,init_checkpoint

from dammethod.trainer import Trainer
from dammethod.tester import Tester

from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train_engine(opt):

    best_error=sys.maxsize

    # init  model
    model=init_model(opt)

    # set criterion and optimizer for model
    criterion,optimizer=setup_model(model,opt)

    writer=True
    # create log dir for tensorboardx
    if writer is not None:
        opt.OTHER.logdir=os.path.join(opt.OTHER.logdir,
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if os.path.exists(opt.OTHER.logdir):
            shutil.rmtree(opt.OTHER.logdir)
        os.makedirs(opt.OTHER.logdir)
        writer = SummaryWriter(opt.OTHER.logdir)

    # set random seed for reduce the randomness
    random.seed(opt.OTHER.seed)
    np.random.seed(opt.OTHER.seed)
    torch.manual_seed(opt.OTHER.seed)

    # reduce the randomness
    cudnn.benchmark = False
    cudnn.deterministic=True


    # resume the training or initmodel
    if opt.TRAIN.resume==True:

        if os.path.isfile(opt.TRAIN.resume_add):
            model, optimizer, best_error, opt = resume_checkpoint(model, optimizer, opt)

        else:
            raise Exception("No such resume file")

    elif opt.TRAIN.initmodel==True:
        # print(opt.TRAIN.initmodel_add)
        if os.path.isfile(opt.TRAIN.initmodel_add):

            model = init_checkpoint(model, opt)

        else:
            raise Exception("No such init model para")

    dataloader=GazeFollowLoader(opt)
    train_loader=dataloader.train_loader
    val_loader=dataloader.val_loader

    # init trainer and validator for gazemodel
    trainer=Trainer(model,criterion,optimizer,train_loader,val_loader,opt,writer=writer)

    tester=Tester(model,criterion,val_loader,opt,writer=writer)

    trainer.get_best_error(best_error)

    optimizer.zero_grad()

    print("Total epoch:{}".format(opt.TRAIN.end_epoch))

    for epoch in range(opt.TRAIN.start_epoch,opt.TRAIN.end_epoch):

        print("Epoch number:{} | Learning rate:{}\n".format(epoch,optimizer.param_groups[0]["lr"]))

        trainer.train(epoch, opt)

        if epoch%opt.TRAIN.save_intervel==0:

          save_checkpoint(model,optimizer,best_error,epoch,opt)

        time.sleep(0.03)

        current_err,_,_=tester.test(opt)

        best_error=min(current_err,best_error)

def test_engine(opt):

    # init model
    model=init_model(opt)

    # set criterion and optimizer for gaze model
    criterion,optimizer=setup_model(model,opt)

    random.seed(opt.OTHER.seed)
    np.random.seed(opt.OTHER.seed)
    torch.manual_seed(opt.OTHER.seed)

    cudnn.deterministic=True

    if opt.TRAIN.resume==True:

        if os.path.isfile(opt.TRAIN.resume_add):
            gazemodel, optimizer, best_error, opt = resume_checkpoint(model, optimizer, opt)

        else:
            raise Exception("No such model file")
    else:
        raise Exception("Please set the model file")

    dataloader = GazeFollowLoader(opt)
    val_loader=dataloader.val_loader

    # init trainer and validator for gazemodel
    tester=Tester(gazemodel,criterion,val_loader,opt,writer=None)

    eval_dist,eval_mindist,eval_auc = tester.test(opt)

    print("Eval Avg dist.: {} | Eval Min dist.:{} | Eval AUC :{}".format(eval_dist,eval_mindist,eval_auc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch Attention Model"
    )
    parser.add_argument(
        "--cfg",
        default="config/gazefollow_cfg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="choose if use gpus"
    )
    parser.add_argument(
        "--is_train",
        action="store_true",
        default=False,
        help="choose if train"
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        default=False,
        help="choose if test"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.OTHER.device='cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    print("The model running on {}".format(cfg.OTHER.device))


    if args.is_train:
        train_engine(cfg)
    elif args.is_test:
        test_engine(cfg)




