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

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


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


def load_state(path, model, optimizer=None, scheduler=None):
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
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

        if optimizer != None:
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                path, start_epoch))
            return best_acc, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(path))


def trans_state(path, model, net_code):
    def map_func(storage, location):
        return storage.cuda()

    checkpoint = torch.load(path, map_location=map_func)
    state_dict = checkpoint['state_dict']
    own_state = model.state_dict()
    # print('own_state:', own_state.keys())
    # print('ckpt state_dict:', state_dict.keys())
    used_keys = []
    for name, param in state_dict.items():
        origin_name = name[:]
        names = name.split('.')
        if names[0] == 'module':
            names = names[1:]
        if len(names) < 3:
            names = names[:]
        elif names[0] == 'stem' or names[2] == 'preprocess0' or names[2] == 'preprocess1':
            names = names[:]
        elif net_code[int(names[3])] == '01':
            names = names[:4] + ['op'] + names[6:]
        elif net_code[int(names[3])] == '11':
            if int(names[5]) == 1:
                names = names[:4] + ['op'] + ['res'] + names[6:]
            else:
                names = names[:4] + ['op'] + ['conv'] + names[6:]
        elif net_code[int(names[3])] == '10' and names[6] != 'op':
            names = names[:4] + ['op'] + names[6:]
        else:
            continue

        name = '.'.join(names)
        if name in own_state:
            used_keys.append(name)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                if own_state[name].size() != param.size():
                    raise
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        else:
            print('unexpected key "{}" in state_dict'
                  .format(name))
            continue
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    missing = set(own_state.keys()) - set(used_keys)
    if len(missing) > 0:
        print(missing)
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def trans_state_free(path, model, net_code):
    def map_func(storage, location):
        return storage.cuda()

    checkpoint = torch.load(path, map_location=map_func)
    state_dict = checkpoint['state_dict']
    own_state = model.state_dict()
    print('own_state:', own_state.keys())
    print('ckpt state_dict:', state_dict.keys())
    used_keys = []
    for name, param in state_dict.items():
        origin_name = name[:]
        names = name.split('.')
        if names[0] == 'module':
            names = names[1:]

        if len(names) < 3:
            names = names[:]
        elif names[0] == 'stem' or names[2] == 'preprocess0' or names[2] == 'preprocess1':
            names = names[:]
        elif net_code[int(names[1])][int(names[3])] == '01':
            names = names[:4] + ['op'] + names[6:]
        elif net_code[int(names[1])][int(names[3])] == '11':
            if int(names[5]) == 1:
                names = names[:4] + ['op'] + ['res'] + names[6:]
            else:
                names = names[:4] + ['op'] + ['conv'] + names[6:]
        elif net_code[int(names[1])][int(names[3])] == '10' and names[6] != 'op':
            names = names[:4] + ['op'] + names[6:]
        else:
            continue

        name = '.'.join(names)
        print(name, origin_name)
        if name in own_state:
            used_keys.append(name)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                if own_state[name].size() != param.size():
                    raise
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        else:
            print('unexpected key "{}" in state_dict'
                  .format(name))
            continue
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    missing = set(own_state.keys()) - set(used_keys)
    if len(missing) > 0:
        print(missing)
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))


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
        # for i in range(len(path.split('/'))):
        #     if path.split('/')[i] == '.':
        #         continue
        #     os.mkdir(path.split('/')[i])
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
