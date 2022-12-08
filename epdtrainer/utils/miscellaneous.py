import os
import sys
from loguru import logger
from easydict import EasyDict
from time import localtime

def get_timetag():
    t = localtime()
    timetag = '%d%.2d%.2d%.2d%.2d%.2d' % (
        t.tm_year,
        t.tm_mon,
        t.tm_mday,
        t.tm_hour,
        t.tm_min,
        t.tm_sec)

    return timetag

def setup_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'execution@{time}.log')
    logger.add(log_path)
    return


def is_img_fn(src_path):
    post = src_path[src_path.rfind('.'):]
    if post.lower() in [
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
    ]:
        return True

    return False


def get_subfiles(path, length=None):
    out_ls = []

    def _func(x, ls=[]):
        for fn in os.listdir(x):
            full_path = os.path.join(x, fn)
            if os.path.isdir(full_path):
                _func(full_path, ls)
            else:
                ls.append(full_path)
            if length != None and len(out_ls) > length:
                return
        return
    _func(path, out_ls)
    # out_ls.sort()
    return out_ls


def update_dict(d, dd):
    """Update dict recursively"""
    if dd is None:
        return
    for k, v in dd.items():
        if isinstance(v, (dict, EasyDict)):
            if k not in d:
                d[k] = EasyDict()
            update_dict(d[k], v)
        else:
            d[k] = v


def load_config(filename):
    from importlib import import_module

    filename = os.path.abspath(os.path.expanduser(filename))
    assert filename.endswith('.py')
    module_name = os.path.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = os.path.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    return mod


def stastics_detail(stastic_dict):
    return ', '.join([f'{k}: {v:>8.5f}\t' for k, v in stastic_dict.items()])
