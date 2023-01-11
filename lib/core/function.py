# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Geonu Lee (lkw3139@gachon.ac.kr)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
from utils.consistency_loss import spatial_consistency, semantic_consistency
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def cal_loss(output, target, weights, reduction='mean'):
    eps = 1e-12
    cur_weights = target * torch.exp(1 - weights) + (1 - target) * torch.exp(weights)
    loss = cur_weights * (target * torch.log(output + eps)) + ((1 - target) * torch.log(1 - output + eps))

    if reduction is 'mean':
        return torch.neg(torch.mean(loss))
    elif reduction is 'sum':
        return torch.neg(torch.mean(loss))
    else:
        return torch.neg(loss)

def moving_average(base, cur, alpha=0.9):
    base = base*alpha + cur*(1-alpha)
    return base

def train(args, config, train_loader, model, pos_weights, memory_spatial, memory_semantic, optimizer, epoch,
          output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    sigmoid = torch.nn.Sigmoid()
    model.train()

    end = time.time()
    gt_list = []
    preds_probs = []
    # for i, (input, target, target_weight, meta) in enumerate(train_loader):
    for i, (imgs, gt_label, imgname) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        outputs, feature, fc_weights = model(imgs)

        # loss
        loss_cls = cal_loss(sigmoid(outputs), gt_label, pos_weights)
        if epoch > 3:
            semantic_loss, memory_semantic = semantic_consistency(args, outputs, gt_label, config.MODEL.NUM_ATTR, feature, fc_weights, memory_semantic)
            spatial_loss, memory_spatial = spatial_consistency(args, outputs, gt_label, config.MODEL.NUM_ATTR, feature, fc_weights, memory_spatial)
            loss = loss_cls + 1.0 * spatial_loss + 0.1 * semantic_loss
        else:
            loss = loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=imgs.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        gt_list.append(gt_label.cpu().numpy())
        out = sigmoid(outputs).detach().cpu().numpy()
        preds_probs.append(out)
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return losses.avg, gt_label, preds_probs, memory_semantic, memory_spatial

def validate(epoch, args, memory_spatial, memory_semantic, config, val_loader, model, pos_weights, output_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    sigmoid = torch.nn.Sigmoid()
    model.eval()

    idx = 0
    gt_list = []
    preds_probs = []
    with torch.no_grad():
        end = time.time()
        for i, (imgs, gt_label, imgname) in enumerate(val_loader):
            # compute output
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            outputs, feature, fc_weights = model(imgs)

            # loss
            loss_cls = cal_loss(sigmoid(outputs), gt_label, pos_weights)
            if epoch > 3:
                semantic_loss, memory_semantic = semantic_consistency(args, outputs, gt_label, config.MODEL.NUM_ATTR,
                                                                      feature, fc_weights, memory_semantic)
                spatial_loss, memory_spatial = spatial_consistency(args, outputs, gt_label, config.MODEL.NUM_ATTR,
                                                                   feature, fc_weights, memory_spatial)
                loss = loss_cls + 1.0 * spatial_loss + 0.1 * semantic_loss
            else:
                loss = loss_cls
            num_images = imgs.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

            gt_list.append(gt_label.cpu().numpy()) #
            out = sigmoid(outputs).detach().cpu().numpy()
            preds_probs.append(out)

        gt_label = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1


    return losses.avg, gt_label, preds_probs, memory_semantic, memory_spatial


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
