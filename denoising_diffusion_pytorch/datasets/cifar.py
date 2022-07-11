import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


cifar10_default_root_dir = "/home/ubuntu/file/data/cifar10"

class MyCIFAR10(Dataset):
    trans0 = transforms.ToTensor()

    def __init__(self,
                 root_dir=cifar10_default_root_dir,
                 download=True):
        self.cifar10 = CIFAR10(root=root_dir,
                               download=download)

    def __getitem__(self, item):
        image, target = self.cifar10[item]
        return self.trans0(image)

    def __len__(self):
        return len(self.cifar10)
