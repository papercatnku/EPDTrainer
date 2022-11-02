import os
from loguru import logger

from collections import OrderedDict, defaultdict
from easydict import EasyDict
import torch
import torch.nn as nn
import torchvision


def load_resume(resume_path):
    logger.info(f'[I] Loading states from {resume_path}')
    return torch.load(resume_path, map_location='cpu')


def create_model(funcs, checkpoint):
    model = funcs.create_network()
    if checkpoint:
        new_state = OrderedDict()
        for k, v in checkpoint['model_state'].items():
            if k.startswith('module'):
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state)
    else:
        import torch.nn.init as tinit
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                tinit.xavier_normal_(m.weight, gain=0.2)
    return model


def create_optimizer(model, checkpoint, config, funcs) -> torch.optim.Optimizer:
    create_optimizer_fn = funcs.get('create_optimizer') or config.optimizer
    weight_decay = config.get('weight_decay', 3e-4)
    if isinstance(create_optimizer_fn, str):
        name = create_optimizer_fn
        lr = config.learn_rate
        if name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)  # 1e-10
        elif name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            assert NotImplementedError(name)
    else:
        optimizer = create_optimizer_fn()
    if checkpoint and config.get('qat', True):
        optimizer_type = checkpoint.get('optimizer')
        if optimizer_type is None or isinstance(optimizer, optimizer_type):
            optimizer_state = checkpoint.get('optimizer_state')
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
    return optimizer


def create_scheduler(optimizer, checkpoint, config, funcs, step_per_epoch=-1) -> torch.optim.lr_scheduler._LRScheduler:
    create_scheduler_fn = funcs.get(
        'create_scheduler', 'onecycle') or config.scheduler
    if create_scheduler_fn is None:
        return None
    if isinstance(create_scheduler_fn, str):
        last_epoch = -1
        if config.get('resume_epoch'):
            last_epoch = config.get('resume_epoch')
        name = create_scheduler_fn
        if name == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.975,
                last_epoch=last_epoch)  # pow(0.99, 300) = 0.04904089407128572
        elif name == 'cos':
            lr = config.learn_rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=2, eta_min=lr * 0.1, last_epoch=last_epoch)
        elif name == 'onecycle':
            # TODO: 当前采用默认值
            # TODO: OneCycleLR自适应resume起点epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                config.learn_rate,
                total_steps=config.epochs,
                last_epoch=last_epoch,
            )
    else:
        scheduler = create_scheduler_fn()
    # if checkpoint and config.qat != True:
    #     scheduler_type = checkpoint.get('scheduler')
    #     if scheduler_type is None or isinstance(scheduler, scheduler_type):
    #         scheduler_state = checkpoint.get('scheduler_state')
    #         if scheduler_state:
    #             scheduler.load_state_dict(scheduler_state)
    return scheduler


def get_saved_model_path(save_dir, epoch, tag=''):
    return os.path.join(save_dir, tag + f'model_{epoch}.pth')


def push_cuda_data(X):
    if isinstance(X, torch.Tensor):
        return X.cuda()
    elif isinstance(X, tuple):
        return (push_cuda_data(x) for x in X)
    elif isinstance(X, list):
        return [push_cuda_data(x) for x in X]
    elif isinstance(X, dict):
        return {k: push_cuda_data(v) for k, v in X.items()}
    else:
        raise TypeError(f'Can not copy data to cuda: {X}')


def get_cuda_data(X):
    if isinstance(X, torch.Tensor):
        return X.cpu()
    elif isinstance(X, tuple):
        return (get_cuda_data(x) for x in X)
    elif isinstance(X, list):
        return [get_cuda_data(x) for x in X]
    elif isinstance(X, dict):
        return {k: get_cuda_data(v) for k, v in X.items()}
    else:
        raise TypeError(f'Can not copy data to cuda: {X}')
