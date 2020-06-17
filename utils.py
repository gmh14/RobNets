import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import shutil
import torch
import logging
import torch
from collections import defaultdict
import numpy as np


def calculate_gates_rate(gates, batch_size):
    gates_rate = []
    for gate in gates:
        gates_rate.append(np.sum(np.squeeze(gate.data.cpu().numpy()), axis=0) / float(batch_size))
    return np.array(gates_rate)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_state(path, model, optimizer=None, scheduler=None, rank=None):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('rank:', rank, 'caution: missing keys from checkpoint {}: {}'.format(path, k))

        if optimizer != None:
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler = checkpoint['scheduler']
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                path, start_epoch))
            return best_acc, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(path))


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def split_net_gate_param(model):
    net_param_group = []
    net_param_name_group = []
    gate_param_group = []
    gate_param_name_group = []
    for name, p in model.named_parameters():
        if not ('gate' in str(name)):
            net_param_group.append(p)
            net_param_name_group.append(name)
        else:
            gate_param_group.append(p)
            gate_param_name_group.append(name)
    return net_param_group, net_param_name_group, gate_param_group, gate_param_name_group


def param_group_no_wd(model, gate_param_name_group=[]):
    pgroup_no_wd = []
    names_no_wd = []
    pgroup_normal = []

    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                if name + '.bias' in gate_param_name_group:
                    continue
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                if name + '.bias' in gate_param_name_group:
                    continue
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d)
              or isinstance(m, torch.nn.BatchNorm1d)):
            if m.weight is not None:
                if name + '.weight' in gate_param_name_group:
                    continue
                pgroup_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                if name + '.bias' in gate_param_name_group:
                    continue
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1

    for name, p in model.named_parameters():
        if not (name in names_no_wd or name in gate_param_name_group):
            pgroup_normal.append(p)

    return [{'params': pgroup_normal}, {'params': pgroup_no_wd, 'weight_decay': 0.0}], type2num, len(
        pgroup_normal), len(pgroup_no_wd)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
