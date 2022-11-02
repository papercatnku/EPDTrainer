import argparse

import os


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)


def get_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Hyper parameters')
    parser.add_argument('--config', '-c', type=file_path,
                        required=True, help='path to config file')
    parser.add_argument('--exp_name', '-n', type=str,
                        default=None, help='experiment name')
    parser.add_argument('--schema', '-s', type=str, default='train',
                        choices=['train', 'export', 'export_torchscript', 'eval'], help='running schema')
    # train utils
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloader')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=None, help='batch size of train dataloader')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='previous saved epoch to restart from')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to previous saved model to restart from')
    parser.add_argument('--eval_ratio', type=str,
                        default=None, help='evaluation data ratio')
    parser.add_argument('--gpus', '-g', type=str, default='7', help='gpu env')
    parser.add_argument('--epochs', '-e', type=int,
                        default=None, help='epochs of training loops')

    # TiDL
    parser.add_argument('--qat_tda4', action='store_true',
                        help='whether do quantization aware training')
    # qat
    parser.add_argument('--qat', action='store_true',
                        help='whether do pytorch quantization aware training')
    args = parser.parse_args(arg_list)
    return args
