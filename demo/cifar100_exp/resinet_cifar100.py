import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from easydict import EasyDict

# from trainer.trainer_static import trainer_static
# trainer = trainer_static
import epdtrainer
from epdtrainer.trainer import trainer_static as trainer
from epdtrainer.decoder.clssification_decoder import clsification_topk_decoder
from epdtrainer.evaluator.eval_clstopk import cls_withtopk_eval
from epdtrainer.visualizer.visualizer_base import visualizer_base

from loss import clsification_loss
from dataloader import create_dataloaders
from net import resne_cls_prototypet


config = EasyDict(
    exp_name='cifar100_resinet',
    schema='train',
    use_cuda=True,
    in_channels=3,
    size=32,
    batch_size=256,
    num_class=10,

    # net input/output adapt auxilary
    input_names=['data', ],
    output_names=['cls_pred', ],
    # resnet params
    layers=[3, 3, ],
    channels_ls=[32, 64, 128],
    strides=[2, 2, ],
    block='bottleneck',
    out_channels=128 * 4,
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
    cifar_type='cifar100'
)

funcs = EasyDict(

    # share in train/eval phase
    create_data_loader=lambda x: create_dataloaders(x),
    create_network=lambda: resne_cls_prototypet(config),
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
