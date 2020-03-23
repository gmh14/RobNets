import numpy as np
import torch.backends.cudnn as cudnn

import sys
import os
import argparse
import pprint

import utils
import logging
import time
import glob
import shutil
from mmcv import Config

import architecture_code
import models
from dataset import dataset_entry
from attack import *

Debug = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='location of the config file')
args = parser.parse_args()


def main():
    global cfg

    cfg = Config.fromfile(args.config)

    cfg.save = '{}/{}-{}-{}'.format(cfg.save_path, cfg.model, cfg.dataset,
                                    time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(cfg.save)

    logger = utils.create_logger('global_logger', cfg.save + '/log.txt')

    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    # Set cuda device & seed
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    # Model
    print('==> Building model..')
    arch_code = eval('architecture_code.{}'.format(cfg.model))
    net = models.model_entry(cfg, arch_code)
    net = net.cuda()

    cfg.netpara = sum(p.numel() for p in net.parameters()) / 1e6
    logger.info('config: {}'.format(pprint.pformat(cfg)))

    # Load checkpoint.
    if not Debug:
        print('==> Resuming from checkpoint..')
        utils.load_state(cfg.resume_path, net)

    # Data
    print('==> Preparing data..')

    testloader = dataset_entry(cfg)
    criterion = nn.CrossEntropyLoss()
    net_adv = AttackPGD(net, cfg.attack_param)

    print('==> Testing on Clean Data..')
    test(net, testloader, criterion)

    print('==> Testing on Adversarial Data..')
    test(net_adv, testloader, criterion, adv=True)


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
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            if batch_idx % cfg.report_freq == 0:
                logger.info(
                    'Test: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    .format(batch_idx, len(testloader), loss=losses, top1=top1, top5=top5))

    final_loss = losses.avg
    final_top1 = top1.avg
    final_top5 = top5.avg

    logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\t'.format(final_top1, final_top5, final_loss))

    return final_top1


if __name__ == '__main__':
    main()
