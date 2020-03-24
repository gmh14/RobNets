import numpy as np
from .basic_operations import *


class ChosenOperation(nn.Module):

    def __init__(self, C, stride, genotype):
        super(ChosenOperation, self).__init__()
        self.op = operation_canditates[genotype](C, stride)

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super(Cell, self).__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        # For search stage, the affine of BN should be set to False, in order to avoid conflict with architecture params
        self.affine = False

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._ops = nn.ModuleList()
        self._complie(C, reduction, genotype)

    def _complie(self, C, reduction, genotype):
        offset = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = ChosenOperation(C, stride, genotype[offset + j])
                self._ops.append(op)
            offset += 2 + i

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, genotype_list, C=36, num_classes=10, layers=20, steps=4, multiplier=4, stem_multiplier=3,
                 share=False, AdPoolSize=1, ImgNetBB=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._share = share
        self._ImgNetBB = ImgNetBB

        if self._ImgNetBB:
            self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C // 2),
                                       nn.ReLU(inplace=False),
                                       nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C))
            self.stem1 = nn.Sequential(nn.ReLU(inplace=False),
                                       nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(C))
            C_prev_prev, C_prev, C_curr = C, C, C
            reduction_prev = True

        else:
            C_curr = stem_multiplier * C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
            reduction_prev = False

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                        genotype=genotype_list[0] if self._share else genotype_list[i])
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        if self._ImgNetBB:
            self.global_pooling = nn.AvgPool2d(7)
        else:
            self.global_pooling = nn.AdaptiveAvgPool2d(AdPoolSize)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        if self._ImgNetBB:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
