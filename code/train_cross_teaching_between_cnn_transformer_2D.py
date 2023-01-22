# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume

import torch
import torch.nn as nn

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='fpn', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    fce_loss=FocalLossMultiClass()
    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model1=xavier_normal_init_weight(model1)
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    # model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    # total_slices = len(db_train)
    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    total_slices=2104
    labeled_slice=212
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.Adam(model1.parameters(), lr=base_lr,
                           weight_decay=0.0001)
    optimizer2 = optim.Adam(model2.parameters(), lr=base_lr, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch3,label_batch2,label_batch1,label_batch0 = sampled_batch['image'], sampled_batch['label3'],sampled_batch['label2'],sampled_batch['label1'],sampled_batch['label0']
            volume_batch, label_batch3,label_batch2,label_batch1,label_batch0 = volume_batch.cuda(), label_batch3.cuda(),label_batch2.cuda(),label_batch1.cuda(),label_batch0.cuda()

            outputs1a,outputs1b,outputs1c,outputs1d = model1(volume_batch)

            outputs_soft1a = torch.softmax(outputs1a, dim=1)
            outputs_soft1b = torch.softmax(outputs1b, dim=1)
            outputs_soft1c = torch.softmax(outputs1c, dim=1)
            outputs_soft1d = torch.softmax(outputs1d, dim=1)

            outputs2a,outputs2b,outputs2c,outputs2d = model2(volume_batch)
            outputs_soft2a = torch.softmax(outputs2a, dim=1)
            outputs_soft2b = torch.softmax(outputs2b, dim=1)
            outputs_soft2c = torch.softmax(outputs2c, dim=1)
            outputs_soft2d = torch.softmax(outputs2d, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss2 = 0.6 * (fce_loss(outputs2c[:args.labeled_bs], label_batch2[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2c[:args.labeled_bs], label_batch2[:args.labeled_bs].unsqueeze(1)))+0.3 * (fce_loss(outputs2b[:args.labeled_bs], label_batch1[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2b[:args.labeled_bs], label_batch1[:args.labeled_bs].unsqueeze(1)))+0.05 * (fce_loss(outputs2a[:args.labeled_bs], label_batch0[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2a[:args.labeled_bs], label_batch0[:args.labeled_bs].unsqueeze(1)))+0.05 * (fce_loss(outputs2d[:args.labeled_bs], label_batch3[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2d[:args.labeled_bs], label_batch3[:args.labeled_bs].unsqueeze(1)))

            loss1 = 0.6 * (fce_loss(outputs1c[:args.labeled_bs], label_batch2[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1c[:args.labeled_bs], label_batch2[:args.labeled_bs].unsqueeze(1)))+0.3 * (fce_loss(outputs1a[:args.labeled_bs], label_batch0[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1a[:args.labeled_bs], label_batch0[:args.labeled_bs].unsqueeze(1)))+0.05 * (fce_loss(outputs1b[:args.labeled_bs], label_batch1[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1b[:args.labeled_bs], label_batch1[:args.labeled_bs].unsqueeze(1)))+0.05 * (fce_loss(outputs1d[:args.labeled_bs], label_batch3[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1d[:args.labeled_bs], label_batch3[:args.labeled_bs].unsqueeze(1)))


            pseudo_outputs1a = torch.argmax(
                outputs_soft1a[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2a = torch.argmax(
                outputs_soft2a[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs1b = torch.argmax(
                outputs_soft1b[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2b = torch.argmax(
                outputs_soft2b[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs1c = torch.argmax(
                outputs_soft1c[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2c = torch.argmax(
                outputs_soft2c[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs1d = torch.argmax(
                outputs_soft1d[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2d = torch.argmax(
                outputs_soft2d[args.labeled_bs:].detach(), dim=1, keepdim=False)


            pseudo_supervision1a = dice_loss(
                outputs_soft1a[args.labeled_bs:], pseudo_outputs2a.unsqueeze(1))
            pseudo_supervision2a = dice_loss(
                outputs_soft2a[args.labeled_bs:], pseudo_outputs1a.unsqueeze(1))

            pseudo_supervision1b = dice_loss(
                outputs_soft1b[args.labeled_bs:], pseudo_outputs2b.unsqueeze(1))
            pseudo_supervision2b = dice_loss(
                outputs_soft2b[args.labeled_bs:], pseudo_outputs1b.unsqueeze(1))

            pseudo_supervision1c = dice_loss(
                outputs_soft1c[args.labeled_bs:], pseudo_outputs2c.unsqueeze(1))
            pseudo_supervision2c = dice_loss(
                outputs_soft2c[args.labeled_bs:], pseudo_outputs1c.unsqueeze(1))

            pseudo_supervision1d = dice_loss(
                outputs_soft1d[args.labeled_bs:], pseudo_outputs2d.unsqueeze(1))
            pseudo_supervision2d = dice_loss(
                outputs_soft2d[args.labeled_bs:], pseudo_outputs1d.unsqueeze(1))

            model1_loss = loss1 + consistency_weight * (pseudo_supervision1a+pseudo_supervision1b+pseudo_supervision1c+pseudo_supervision1d)
            model2_loss = loss2 + consistency_weight * (pseudo_supervision2a+pseudo_supervision2c+pseudo_supervision2b+pseudo_supervision2d)



            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            if iter_num%1000==0:
              
              logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                  iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1d, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2d, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch3[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # if iter_num > 0 and iter_num % 200 == 0:
            #     model1.eval()
            #     metric_list = 0.0
            #     for i_batch, sampled_batch in enumerate(valloader):
            #         metric_i = test_single_volume(
            #             sampled_batch["image"], sampled_batch["label3"], model1, classes=num_classes, patch_size=args.patch_size)
            #         metric_list += np.array(metric_i)
            #     metric_list = metric_list / len(db_val)
            #     for class_i in range(num_classes-1):
            #         writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
            #                           metric_list[class_i, 0], iter_num)
            #         writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
            #                           metric_list[class_i, 1], iter_num)
            #
            #     performance1 = np.mean(metric_list, axis=0)[0]
            #
            #     mean_hd951 = np.mean(metric_list, axis=0)[1]
            #     writer.add_scalar('info/model1_val_mean_dice',
            #                       performance1, iter_num)
            #     writer.add_scalar('info/model1_val_mean_hd95',
            #                       mean_hd951, iter_num)
            #
            #     if performance1 > best_performance1:
            #         best_performance1 = performance1
            #         save_mode_path = os.path.join(snapshot_path,
            #                                       'model1_iter_{}_dice_{}.pth'.format(
            #                                           iter_num, round(best_performance1, 4)))
            #         save_best = os.path.join(snapshot_path,
            #                                  '{}_best_model1.pth'.format(args.model))
            #         torch.save(model1.state_dict(), save_mode_path)
            #         torch.save(model1.state_dict(), save_best)
            #
            #     logging.info(
            #         'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
            #     model1.train()
            #
            #     model2.eval()
            #     metric_list = 0.0
            #     for i_batch, sampled_batch in enumerate(valloader):
            #         metric_i = test_single_volume(
            #             sampled_batch["image"], sampled_batch["label3"
            #                                                   ""], model2, classes=num_classes, patch_size=args.patch_size)
            #         metric_list += np.array(metric_i)
            #     metric_list = metric_list / len(db_val)
            #     for class_i in range(num_classes-1):
            #         writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
            #                           metric_list[class_i, 0], iter_num)
            #         writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
            #                           metric_list[class_i, 1], iter_num)
            #
            #     performance2 = np.mean(metric_list, axis=0)[0]
            #
            #     mean_hd952 = np.mean(metric_list, axis=0)[1]
            #     writer.add_scalar('info/model2_val_mean_dice',
            #                       performance2, iter_num)
            #     writer.add_scalar('info/model2_val_mean_hd95',
            #                       mean_hd952, iter_num)
            #
            #     if performance2 > best_performance2:
            #         best_performance2 = performance2
            #         save_mode_path = os.path.join(snapshot_path,
            #                                       'model2_iter_{}_dice_{}.pth'.format(
            #                                           iter_num, round(best_performance2, 4)))
            #         save_best = os.path.join(snapshot_path,
            #                                  '{}_best_model2.pth'.format(args.model))
            #         torch.save(model2.state_dict(), save_mode_path)
            #         torch.save(model2.state_dict(), save_best)
            #
            #     logging.info(
            #         'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
            #     model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
