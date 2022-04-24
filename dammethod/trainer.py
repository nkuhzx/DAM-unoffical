import torch
import torch.nn as nn
import numpy as np
from dammethod.utils.utils import AverageMeter, MovingAverageMeter, euclid_dist,auc
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, criterion, optimizer, trainloader, valloader, opt, writer=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.trainloader = trainloader
        self.valloader = valloader

        # record train loss
        self.losses = MovingAverageMeter()
        self.l2losses=MovingAverageMeter()
        self.langlosses=MovingAverageMeter()


        self.train_dist=MovingAverageMeter()

        # for eval in valid set
        self.eval_dist = AverageMeter()
        self.eval_mindist = AverageMeter()
        self.eval_angle=AverageMeter()
        self.eval_ap = AverageMeter()

        self.device = opt.OTHER.device
        self.device = torch.device(self.device)

        self.opt = opt
        self.writer = writer

    def get_best_error(self, bs_error):

        self.best_error = bs_error

    def train(self, epoch, opt):

        self.model.train()

        # reset recoder
        self.losses.reset()

        self.eval_dist.reset()
        self.eval_mindist.reset()
        self.eval_ap.reset()

        loader_capacity = len(self.trainloader)
        pbar = tqdm(total=loader_capacity)
        for i, data in enumerate(self.trainloader, 0):

            # for key,value in data.items():
            #     print(key,torch.isnan(value).any(),torch.isinf(value).any())

            self.optimizer.zero_grad()

            opt.OTHER.global_step = opt.OTHER.global_step + 1

            x_img,x_mmimg, x_face, x_leyeimg,x_reyeimg = data["img"],data["mmimg"], data["face"], data["l_eyeimg"],data["r_eyeimg"]

            x_ind,x_gzfield=data["indicator"],data["gazefield"]


            heatmap = data["gaze_heatmap"]
            gazevector=data["gaze_vector"]
            in_out = data["gaze_inside"]
            gaze_value = data["gaze_label"]

            x_img = x_img.to(self.device)
            x_mmimg=x_mmimg.to(self.device)
            x_face=x_face.to(self.device)
            x_leyeimg=x_leyeimg.to(self.device)
            x_reyeimg=x_reyeimg.to(self.device)

            x_gzfield=x_gzfield.to(self.device)
            x_ind=x_ind.to(self.device)

            y_heatmap = heatmap.to(self.device)
            y_gazevector=gazevector.to(self.device)
            y_in_out = in_out.to(self.device).to(torch.float)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            bs = x_img.size(0)

            outs = self.model(x_img,x_mmimg,x_gzfield, x_face,x_leyeimg,x_reyeimg,x_ind)

            pred_gheatmap = outs['heatmap']
            pred_gheatmap = pred_gheatmap.squeeze()

            pred_gvector=outs["gazevector"]

            # heatmap loss
            l2_loss = self.criterion[0](pred_gheatmap, y_heatmap)

            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)

            # l2_loss = torch.mul(l2_loss, y_in_out)
            l2_loss = torch.sum(l2_loss) / bs

            # angle loss
            lang_loss=1-self.criterion[1](pred_gvector,y_gazevector)
            lang_loss=torch.sum(lang_loss)/bs

            total_loss = l2_loss*10000+100*lang_loss
            # print(l2_loss,lang_loss,total_loss)

            total_loss.backward()
            self.optimizer.step()

            self.losses.update(total_loss.item())
            self.l2losses.update(l2_loss.item())
            self.langlosses.update(lang_loss.item())

            if i % opt.OTHER.lossrec_every == 0:
                self.writer.add_scalar("Train_TotalLoss", total_loss.item(), global_step=opt.OTHER.global_step)

                pred_gheatmap = pred_gheatmap.squeeze(1)
                pred_gheatmap = pred_gheatmap.data.cpu().numpy()
                distrain_avg = euclid_dist(pred_gheatmap, gaze_value, type='avg')
                self.train_dist.update(distrain_avg)



            if (i % opt.OTHER.evalrec_every == 0 and i > 0):

                self.valid(self.opt,epoch)

                self.writer.add_scalar("Val_avg_dist", self.eval_dist.avg, global_step=opt.OTHER.global_step)
                self.writer.add_scalar("Val_min_dist", self.eval_mindist.avg, global_step=opt.OTHER.global_step)
                self.writer.add_scalar("Val_ap", self.eval_ap.avg, global_step=opt.OTHER.global_step)

            pbar.set_description("Epoch: [{0}]".format(epoch))
            pbar.set_postfix(eval_avgdist=self.eval_dist.avg,
                             eval_mindist=self.eval_mindist.avg,
                             eval_angle=self.eval_angle.avg,
                             eval_ap=self.eval_ap.avg,
                             loss=self.losses.avg,
                             l2_loss=self.l2losses.avg,
                             ang_loss=self.langlosses.avg,
                             train_dist=self.train_dist.avg,
                             learning_rate=self.optimizer.param_groups[0]["lr"])

            pbar.update(1)

        pbar.close()

    @torch.no_grad()
    def valid(self, epoch, opt):

        self.model.eval()

        self.eval_dist.reset()
        self.eval_mindist.reset()
        self.eval_ap.reset()

        for i, data in enumerate(self.valloader, 0):


            x_img, x_mmimg, x_face, x_leyeimg, x_reyeimg = data["img"], data["mmimg"], data["face"], data["l_eyeimg"], \
                                                           data["r_eyeimg"]

            x_ind, x_gzfield = data["indicator"], data["gazefield"]

            gazevector=data["gaze_vector"]
            gaze_value = data["gaze_label"]

            img_size=data["img_size"]

            x_img = x_img.to(self.device)
            x_mmimg = x_mmimg.to(self.device)
            x_face = x_face.to(self.device)
            x_leyeimg = x_leyeimg.to(self.device)
            x_reyeimg = x_reyeimg.to(self.device)

            x_gzfield = x_gzfield.to(self.device)
            x_ind = x_ind.to(self.device)

            bs = x_img.size(0)
            outs = self.model(x_img,x_mmimg,x_gzfield, x_face,x_leyeimg,x_reyeimg,x_ind)


            pred_heatmap = outs['heatmap']
            pred_heatmap = pred_heatmap.squeeze(1)
            pred_heatmap = pred_heatmap.data.cpu().numpy()

            gt_gazevector=gazevector.to(self.device)
            pred_gazevector=outs["gazevector"]
            # pred_gazevector=pred_gazevector.squeeze()
            # pred_gazevector=pred_gazevector.cpu().numpy()

            distval = euclid_dist(pred_heatmap, gaze_value, type='avg')
            mindistval=euclid_dist(pred_heatmap, gaze_value, type='min')

            cosine_value=self.criterion[1](pred_gazevector,gt_gazevector)
            cosine_value=torch.sum(cosine_value)/bs

            cosine_value=cosine_value.cpu().detach().cpu().numpy()
            # cosine_value=pred_gazevector*gt_gazevector
            # cosine_value=np.sum(cosine_value,axis=1)
            # cosine_value=np.sum(cosine_value)/bs

            # auc_score=auc(gaze_value.numpy(),pred_heatmap,img_size.numpy())

            self.eval_dist.update(distval,bs)
            self.eval_mindist.update(mindistval,bs)
            self.eval_angle.update(cosine_value,bs)
            # self.eval_ap.update(auc_score,bs)


        self.model.train()