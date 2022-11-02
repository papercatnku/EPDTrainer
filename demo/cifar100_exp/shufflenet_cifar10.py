import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from easydict import EasyDict

# from trainer.trainer_static import trainer_static
# trainer = trainer_static
from trainer import trainer_static as trainer
from decoder.clssification_decoder import clsification_topk_decoder
from evaluator.eval_clstopk import cls_withtopk_eval
from visualizer.visualizer_base import visualizer_base

from loss import clsification_loss
from dataloader import create_dataloaders
from net import creat_cls_inf_net


config = EasyDict(
    exp_name='cifar10_shufflenet',
    schema='train',
    use_cuda=True,
    in_channels=3,
    size=32,
    batch_size=256,
    num_class=10,

    # net input/output adapt auxilary
    input_names=['data', ],
    output_names=['cls_pred', ],
    # shufflenet params
    out_channels=256,
    stage_repeats=[4, 8, 4],
    stage_channels=[24, 116, 232, 464],
    #
    eval_topk=1,
    epochs=200,
    # export utitlities
    inf_wh=(32, 32),

    #
    sw=EasyDict(
        num_log_per_epoch=5,
        show_lr=True,
        batch_step=0,
        batch_viz_step=0,
        show_graph=True
    ),
    # optimizer
    optimizer='adam',
    scheduler='onecycle',
    schedule_step_phase='epoch',  # [batch|epoch]

    # miscellaneous
    cifar_download_root='./exp/cifar',
    cifar_type='cifar10'
)

funcs = EasyDict(

    # share in train/eval phase
    create_data_loader=lambda x: create_dataloaders(x),
    create_network=lambda: creat_cls_inf_net(config),
    create_losses=lambda x: clsification_loss(),

    # desgined to be different in train/eval phase
    create_decoder=lambda x: (
        None, clsification_topk_decoder(
            topk=x.eval_topk)),
    create_evaluator=lambda x: (
        None,
        cls_withtopk_eval(nm='val', topk=x.eval_topk)),
    # optional visualize for train & eval phase
    creater_visualizer=lambda x: (
        None, None
    )


)
