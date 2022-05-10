import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import argparse

from gazeestimation.gaze360 import Gaze360Loader
from gazeestimation.utils.model_utils import init_model,setup_model,save_checkpoint,resume_checkpoint
from gazeestimation.trainer import Trainer
from gazeestimation.tester import Tester

from tqdm import tqdm


def train_engine(opt):

    best_val=-sys.maxsize-1

    # setup the model
    gazedirmodel=init_model(opt)

    criterion,optimizer=setup_model(gazedirmodel,opt)

    # set random seed for reduce the randomness
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    cudnn.benchmark=False
    cudnn.deterministic=True


    # init the dataloader
    gazedataloader=Gaze360Loader(opt)

    # for training
    train_loader=gazedataloader.train_loader
    val_loader=gazedataloader.val_loader

    # for test
    test_loader=gazedataloader.test_loader

    trainer=Trainer(gazedirmodel,criterion,optimizer,train_loader,val_loader,opt)
    tester=Tester(gazedirmodel,criterion,test_loader,opt)

    tqdm.write("|Training process Total epoch: {}|".format(opt.end_epoch))
    for epoch in range(opt.end_epoch):

        tqdm.write("|Epoch number: {}|".format(epoch))

        trainer.train(epoch)

        valintest_val=tester.test(epoch,best_val)

        if epoch%opt.save_interval==0 and valintest_val>best_val:

            save_checkpoint(gazedirmodel,optimizer,best_val,epoch,opt)

        best_val=max(best_val,valintest_val)
        tqdm.write("|Error after Epoch {}:{}| Best Error {}".format(epoch,valintest_val,best_val))


def test_engine(opt):

    tqdm.write("|Test process|")
    # setup the model
    gazedirmodel=init_model(opt)

    criterion,optimizer=setup_model(gazedirmodel,opt)

    # set random seed for reduce the randomness
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    cudnn.benchmark=False
    cudnn.deterministic=True


    # init the dataloader
    gazedataloader=Gaze360Loader(opt)

    test_loader=gazedataloader.test_loader

    tester=Tester(gazedirmodel,criterion,test_loader,opt)

    if os.path.exists(opt.model_path):
        gazemodel, optimizer, _ = resume_checkpoint(gazedirmodel, optimizer, opt)
    else:
        raise NotImplemented

    cosine_eval = tester.test(0,0)

    tqdm.write("|eval cosine similiarity: {}|".format(cosine_eval))


if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="../datasets/Gaze360/imgs", help="path to train csv", type=str,metavar="FILE")
    parser.add_argument("--train_csv", default="../datasets/Gaze360_annotations/train_eye.txt", help="path to train csv", type=str,metavar="FILE")
    parser.add_argument("--val_csv", default="../datasets/Gaze360_annotations/val_eye.txt", help="path to validation csv", type=str, metavar="FILE")
    parser.add_argument("--test_csv",default="../datasets/Gaze360_annotations/test_eye.txt", help="path to test csv", type=str, metavar="FILE")

    parser.add_argument("--num_worker",default=12,help="choose the num_worker",type=int)
    parser.add_argument("--seed",default=1234,help="choose the random seed",type=int)

    parser.add_argument("--pretrained",default=True,help="pretrain on Imagenet",type=bool)

    parser.add_argument("--train_batch_size",default=64,help="choose the batch size for training",type=int)
    parser.add_argument("--val_batch_size",default=64,help="choose the batch size for validation",type=int)
    parser.add_argument("--test_batch_size",default=64,help="choose the batch size for test",type=int)

    parser.add_argument("--end_epoch",default=4,help="choose the batch size for test",type=int)

    parser.add_argument("--criterion_train",default="mixed",help="choose the criterion for train",type=str)

    parser.add_argument("--criterion_eval",default="cosinesim",help="choose the criterion",type=str)
    parser.add_argument("--optimizer",default="adam",help="choose the optimizer",type=str)

    parser.add_argument("--lr",default=1e-4,help="choose the learning rate",type=int)

    parser.add_argument("--evalrec",default=400,help="choose the eval times for record",type=int)

    parser.add_argument("--save_interval",default=1,help="choose the eval interval for record",type=int)
    parser.add_argument("--store",default="./checkpoints",help="choose the dir to store model parameters",type=str)

    parser.add_argument("--model_path",default="./checkpoints/gaze360_5epoch.pth.tar",help="choose the dir to store model parameters",type=str)


    parser.add_argument("--input_size",default=224,help="choose the input size for model",type=int)

    parser.add_argument("--gpu",action="store_true",default=True,help="choose to use gpu or not")
    parser.add_argument("--device",default="cuda:0",help="choose to use gpu or not",type=str)

    parser.add_argument("--is_train",action="store_true",default=True,help="choose train or not")

    args=parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        args.device="cuda:0"
    else:
        args.device="cpu"

    print("Device: {}".format(args.device))

    if args.is_train:

        train_engine(args)
    else:
        test_engine(args)

