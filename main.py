import numpy as np
import torch.backends.cudnn as cudnn

import sys
import os
import argparse
import pprint

import logging
import time
import glob
import shutil
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info

import architecture_code
import models
from dataset import dataset_entry
from attack import *
import utils
import lr_scheduler

Debug = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/RobNet_free_cifar10/config.py',
                    help='location of the config file')
parser.add_argument('--distributed', action='store_true', default=False, help='Distributed training')
parser.add_argument('--eval_only', action='store_true', default=False, help='Only evaluate')
parser.set_defaults(augment=True)
args = parser.parse_args()


def main():
    global cfg, rank, world_size

    cfg = Config.fromfile(args.config)

    # Set seed
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    # Model
    print('==> Building model..')
    arch_code = eval('architecture_code.{}'.format(cfg.model))
    net = models.model_entry(cfg, arch_code)
    rank = 0 # for non-distributed
    world_size = 1 # for non-distributed
    if args.distributed:
        print('==> Initializing distributed training..')
        init_dist(launcher='slurm', backend='nccl') # Only support slurm for now, if you would like to personalize your launcher, please refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
        rank, world_size = get_dist_info()
    net = net.cuda()

    cfg.netpara = sum(p.numel() for p in net.parameters()) / 1e6

    start_epoch = 0
    best_acc = 0
    # Load checkpoint.
    if cfg.get('resume_path', False):
        print('==> Resuming from {}checkpoint {}..'.format(('original ' if cfg.resume_path.origin_ckpt else ''), cfg.resume_path.path))
        if cfg.resume_path.origin_ckpt:
            utils.load_state(cfg.resume_path.path, net, rank=rank)
        else:
            if args.distributed:
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device())
            utils.load_state(cfg.resume_path.path, net, rank=rank)

    # Data
    print('==> Preparing data..')
    if args.eval_only:
        testloader = dataset_entry(cfg, args.distributed, args.eval_only)
    else:
        trainloader, testloader, train_sampler, test_sampler = dataset_entry(cfg, args.distributed, args.eval_only)
    criterion = nn.CrossEntropyLoss()
    if not args.eval_only:
        cfg.attack_param.num_steps = 7
    net_adv = AttackPGD(net, cfg.attack_param)

    if not args.eval_only:
        # Train params
        print('==> Setting train parameters..')
        train_param = cfg.train_param
        epochs = train_param.epochs
        init_lr = train_param.learning_rate
        if train_param.get('warm_up_param', False):
            warm_up_param = train_param.warm_up_param
            init_lr = warm_up_param.warm_up_base_lr
            epochs += warm_up_param.warm_up_epochs
        if train_param.get('no_wd', False):
            param_group, type2num, _, _ = utils.param_group_no_wd(net)
            cfg.param_group_no_wd = type2num
            optimizer = torch.optim.SGD(param_group, lr=init_lr, momentum=train_param.momentum, weight_decay=train_param.weight_decay)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=train_param.momentum, weight_decay=train_param.weight_decay)

        scheduler = lr_scheduler.CosineLRScheduler(optimizer, epochs, train_param.learning_rate_min, init_lr, train_param.learning_rate, (warm_up_param.warm_up_epochs if train_param.get('warm_up_param', False) else 0))
    # Log
    print('==> Writing log..')
    if rank == 0:
        cfg.save = '{}/{}-{}-{}'.format(cfg.save_path, cfg.model, cfg.dataset,
                                    time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(cfg.save)
        logger = utils.create_logger('global_logger', cfg.save + '/log.txt')
        logger.info('config: {}'.format(pprint.pformat(cfg)))

    # Evaluation only
    if args.eval_only:
        assert cfg.get('resume_path', False), 'Should set the resume path for the eval_only mode'
        print('==> Testing on Clean Data..')
        test(net, testloader, criterion)
        print('==> Testing on Adversarial Data..')
        test(net_adv, testloader, criterion, adv=True)
        return

    # Training process
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        scheduler.step()
        if rank == 0:
            logger.info('Epoch %d learning rate %e', epoch, scheduler.get_lr()[0])
        
        # Train for one epoch
        train(net_adv, trainloader, criterion, optimizer)

        # Validate for one epoch
        valid_acc = test(net_adv, testloader, criterion, adv=True)

        if rank == 0:
            logger.info('Validation Accuracy: {}'.format(valid_acc))
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
            print('==> Saving')
            state = {'epoch': epoch,
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'state_dict': net.state_dict(),
                     'scheduler': scheduler} 
            utils.save_checkpoint(state, is_best, os.path.join(cfg.save))


def train(net, trainloader, criterion, optimizer):
    objs = utils.AverageMeter(cfg.report_freq)
    top1 = utils.AverageMeter(cfg.report_freq)
    top5 = utils.AverageMeter(cfg.report_freq)
    
    logger = logging.getLogger('global_logger')

    for step, (inputs, targets) in enumerate(trainloader):
        net.train()
        num = inputs.size(0)
        inputs, targets = inputs.cuda(), targets.cuda()
        
        logits, _ = net(inputs, targets)
        loss = criterion(logits, targets)
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reduced_loss = loss.clone() / world_size
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size
        if args.distributed:
            torch.distributed.all_reduce(reduced_loss)
            torch.distributed.all_reduce(reduced_prec1)
            torch.distributed.all_reduce(reduced_prec5)
        objs.update(reduced_loss.clone().item())
        top1.update(reduced_prec1.clone().item())
        top5.update(reduced_prec5.clone().item())

        if step % cfg.report_freq == 0 and rank == 0:
            logger.info('Iter: [{0}/{1}]\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'
                        .format(step, len(trainloader), loss=objs, top1=top1, top5=top5))

        
def test(net, testloader, criterion, adv=False):
    losses = utils.AverageMeter(0)
    top1 = utils.AverageMeter(0)
    top5 = utils.AverageMeter(0)

    logger = logging.getLogger('global_logger')

    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            if not adv:
                outputs = net(inputs)
            else:
                outputs, inputs_adv = net(inputs, targets)

            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, 5))

            num = inputs.size(0)
            losses.update(loss.clone().item(), num)
            top1.update(prec1.clone().item(), num)
            top5.update(prec5.clone().item(), num)
            
            if batch_idx % cfg.report_freq == 0 and rank == 0:
                logger.info(
                    'Test: [{0}/{1}]\t'
		    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
		    'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
		    'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\t'
                    .format(batch_idx, len(testloader), loss=losses, top1=top1, top5=top5))

    final_loss_sum = torch.Tensor([losses.sum]).cuda()
    final_top1_sum = torch.Tensor([top1.sum]).cuda()
    final_top5_sum = torch.Tensor([top5.sum]).cuda()
    total_num = torch.Tensor([losses.count]).cuda()
    if args.distributed:
        torch.distributed.all_reduce(final_loss_sum)
        torch.distributed.all_reduce(final_top1_sum)
        torch.distributed.all_reduce(final_top5_sum)
        torch.distributed.all_reduce(total_num)
    final_loss = final_loss_sum.item() / total_num.item()
    final_top1 = final_top1_sum.item() / total_num.item()
    final_top5 = final_top5_sum.item() / total_num.item()

    logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\t'.format(final_top1, final_top5, final_loss))

    return final_top1


if __name__ == '__main__':
    main()
