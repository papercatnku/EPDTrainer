import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from data.collate import collate_list2dict
# import epdtrainer
# from data.collate import collate_list2dict
# TODO: ddp supported dl


def create_dataloaders(config):
    batch_size = config.get('batch_size', 64)
    train_trans = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=1.0)
        ]
    )
    eval_trans = transforms.Compose(
        [
            transforms.Normalize(mean=0.0, std=1.0)
        ]
    )

    cifar100_root = config.get('cifar_download_root', './exp')

    if config.cifar_type == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=cifar100_root,
            train=True,
            download=True,
            transform=train_trans
        )
        eval_set = torchvision.datasets.CIFAR10(
            root=cifar100_root,
            train=False,
            download=False,
            transform=train_trans
        )
    elif config.cifar_type == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=cifar100_root,
            train=True,
            download=True,
            transform=train_trans
        )
        eval_set = torchvision.datasets.CIFAR100(
            root=cifar100_root,
            train=False,
            download=False,
            transform=train_trans
        )

    collate_func = collate_list2dict(
        names=[
            'data', 'label'
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_list2dict())
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_list2dict()
    )
    return train_loader, eval_loader


if __name__ == '__main__':
    from easydict import EasyDict
    config = EasyDict(
        batch_size=32
    )
    train_dl, eval_dl = create_dataloaders(config)
    for batch in train_dl:
        break
    # print(len(train_dl))
    print(batch)
    pass
    # torchvision.models.ShuffleNetV2
