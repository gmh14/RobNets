import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import io


def dataset_entry(cfg):
    return globals()[cfg.dataset](**cfg.dataset_param)


def cifar10(data_root, batch_size, num_workers):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader


def SVHN(data_root, batch_size, num_workers):
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


def ImgNet(data_root, resize_size, input_img_size, batch_size, num_workers):
    valset = torchvision.datasets.ImageFolder(data_root, transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_img_size),
        transforms.ToTensor(),
    ]))
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return valloader
