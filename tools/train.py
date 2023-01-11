# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Geonu Lee (lkw3139@gachon.ac.kr)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from dataset.augmentation import get_transform
from dataset.pedes_attr.pedes import PedesAttr
import dataset
import models
from metrics.pedestrian_metrics import get_pedestrian_metrics
from easydict import EasyDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--gpu',
                        help='gpu id',
                        required=True,
                        type=str)

    parser.add_argument('--savename',
                        help='experiment save name',
                        required=True,
                        type=str)

    parser.add_argument('--thres',
                        help='semantic loss parameter',
                        required=False,
                        default=0.9,
                        type=float)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.savename, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("run_deterministic")
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    model = eval('models.'+cfg.MODEL.NAME+'.get_par_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    width, height = image_size[0], image_size[1]
    train_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = PedesAttr(cfg=cfg, split='trainval', transform=train_transform)
    valid_set = PedesAttr(cfg=cfg, split='test', transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_result = EasyDict()
    best_epoch = 0
    optimizer = get_optimizer(cfg, model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    model.cuda()

    ## for ssc WL, calculate positive ratio
    total_samples = len(train_set)
    pos_weights = train_set.samples_num / total_samples
    pos_weights = torch.Tensor(pos_weights).cuda()

    ## for SSC regularization
    memory_spatial = torch.zeros([cfg.MODEL.NUM_ATTR, 49]).cuda()
    memory_spatial_valid = torch.zeros([cfg.MODEL.NUM_ATTR, 49]).cuda()
    memory_semantic = torch.zeros([cfg.MODEL.NUM_ATTR, 2048]).cuda()
    memory_semantic_valid = torch.zeros([cfg.MODEL.NUM_ATTR, 2048]).cuda()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        train_loss, train_gt, train_probs, memory_semantic, memory_spatial = train(args, cfg, train_loader, model, pos_weights, memory_spatial, memory_semantic, optimizer, epoch,
              final_output_dir, writer_dict)

        # evaluate on validation set
        valid_loss, valid_gt, valid_probs, memory_semantic_valid, memory_spatial_valid = validate(epoch, args, memory_spatial_valid, memory_semantic_valid,
            cfg, valid_loader, model, pos_weights, final_output_dir, writer_dict
        )
        
        lr_scheduler.step(valid_loss)

        train_result = get_pedestrian_metrics(train_gt, train_probs, index=None)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None)


        msg_train_loss = f'Evaluation on train set, train losses {train_loss}'
        msg_train_results_0 = 'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
                  train_result.ma, np.mean(train_result.label_f1),
                  np.mean(train_result.label_pos_recall),
                  np.mean(train_result.label_neg_recall))
        msg_train_results_1 = 'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                  train_result.instance_f1)

        msg_valid_loss = f'Evaluation on test set, valid losses {valid_loss}'
        msg_valid_results_0 = 'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
                  valid_result.ma, np.mean(valid_result.label_f1),
                  np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall))
        msg_valid_results_1 = 'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1)

        logger.info(msg_train_loss)
        logger.info(msg_train_results_0)
        logger.info(msg_train_results_1)
        logger.info(msg_valid_loss)
        logger.info(msg_valid_results_0)
        logger.info(msg_valid_results_1)

        perf_indicator = valid_result.instance_f1
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
            best_result = valid_result
            best_epoch = epoch
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'train_global_steps': writer_dict['train_global_steps'],
            'valid_global_steps': writer_dict['valid_global_steps'],
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    logger.info("best_epoch:{}".format(best_epoch))
    msg_valid_results_0 = 'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
        best_result.ma, np.mean(best_result.label_f1),
        np.mean(best_result.label_pos_recall),
        np.mean(best_result.label_neg_recall))
    msg_valid_results_1 = 'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
        best_result.instance_acc, best_result.instance_prec, best_result.instance_recall,
        best_result.instance_f1)
    logger.info(msg_valid_results_0)
    logger.info(msg_valid_results_1)




if __name__ == '__main__':
    main()
