import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import io
import utils
from mmcv.runner import get_dist_info


def dataset_entry(cfg, distributed):
    return globals()[cfg.dataset](distributed=distributed, **cfg.dataset_param)


def cifar10(data_root, batch_size, num_workers, distributed, cutout=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
    ])
    if cutout:
        transform_train.transforms.append(utils.Cutout(n_holes=1, length=16))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_sampler = None
    test_sampler = None
    if distributed:
        rank, world_size = get_dist_info()
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=world_size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, shuffle=(train_sampler is None))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    return trainloader, testloader, train_sampler, test_sampler


def SVHN(data_root, batch_size, num_workers, **kwargs):
    class SVHN_Dataset(torch.utils.data.Dataset):

        training_file = 'train_32x32.mat'
        test_file = 'test_32x32.mat'

        def __init__(self, train=True, transform=None, target_transform=None):
            self.transform = transform
            self.target_transform = target_transform
            self.train = train

            if self.train:
                data_file = self.training_file
            else:
                data_file = self.test_file

            import scipy.io as sio
            loaded_mat = sio.loadmat(os.path.join(data_root, data_file))

            self.data = loaded_mat['X']
            self.targets = loaded_mat['y'].astype(np.int64).squeeze()

            # SVHN assigns the class label "10" to the digit 0
            # change the class labels to be in the range [0, C-1]
            np.place(self.targets, self.targets == 10, 0)
            self.data = np.transpose(self.data, (3, 2, 0, 1))

        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        def __len__(self):
            return len(self.data)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = SVHN_Dataset(False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader


def ImgNet(data_root, resize_size, input_img_size, batch_size, num_workers, **kwargs):
    valset = torchvision.datasets.ImageFolder(data_root, transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_img_size),
        transforms.ToTensor(),
    ]))
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return valloader
