import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

from gazeestimation.gazedirmodel import GazeDirmodel

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1 :
        n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        m.weight.data.normal_(0,math.sqrt(2./n))

    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.zero_()


def init_model(opt):

    model=GazeDirmodel()

    model.head_pos.fc1.apply(weights_init)
    model.head_pos.fc2.apply(weights_init)

    model.fc1.apply(weights_init)
    model.fc2.apply(weights_init)
    # model.fc2.apply(weights_init)

    model.to(opt.device)

    return model

def setup_model(model,opt):

    if opt.criterion_eval=="cosinesim":

        criterion_eval=nn.CosineSimilarity()
    else:
        raise NotImplemented


    if opt.criterion_train=="l1":

        criterion_train=nn.L1Loss()
    elif opt.criterion_train=="mixed":
        criterion_train = [nn.L1Loss(), nn.CosineSimilarity()]
    else:

        raise NotImplemented



    criterion=[criterion_train,criterion_eval]

    if opt.optimizer=="adam":

        optimizer=optim.Adam(params=model.parameters(),
                             lr=opt.lr)

    else:
        raise NotImplemented

    return criterion,optimizer


def save_checkpoint(model,optimizer,best_error,epoch,opt):

    cur_state={
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_error':best_error,
        'optimizer':optimizer.state_dict()
    }

    save_filename='gaze360'+'_'+str(epoch)+'epoch.pth.tar'

    if os.path.exists(opt.store)==False:

        os.makedirs(opt.store)

    torch.save(cur_state,os.path.join(opt.store,save_filename) )


def resume_checkpoint(model,optimizer,opt):

    checkpoint=torch.load(opt.model_path)

    best_val=checkpoint['best_error']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> Loading checkpoint '{}' ".format(opt.model_path))

    return model,optimizer,best_val



