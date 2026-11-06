from pathlib import Path
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

IMAGE_SIZE = (28, 28)
input_size = 28*28


def data_loaders(
        path:Path,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        download: bool = True
) -> tuple[DataLoader, DataLoader]:
    transfrom_ = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.bernoulli)])
    train=torchvision.datasets.MNIST(str(path), train=True, transform=transfrom_,download=download)
    test=torchvision.datasets.MNIST(str(path), train=False,transform=transfrom_,download=download)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train, shuffle=True, drop_last=True, **common),
        DataLoader(test, shuffle=False, drop_last=False,**common)
    )

def data_gan_loaders(
        path:Path,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        download: bool = True
) -> tuple[DataLoader, DataLoader]:
    transfrom_ = transforms.Compose([transforms.ToTensor()])
    train=torchvision.datasets.MNIST(str(path), train=True, transform=transfrom_,download=download)
    test=torchvision.datasets.MNIST(str(path), train=False,transform=transfrom_,download=download)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train, shuffle=True, drop_last=True, **common),
        DataLoader(test, shuffle=False, drop_last=False,**common)
    )