from args import get_args
from epdtrainer.utils.miscellaneous import load_config


def main(args):
    mod = load_config(args.config)
    trainer = mod.trainer
    trainer = trainer(args)
    trainer.run()
    return


if __name__ == '__main__':
    import os
    arg_list = None
    args = get_args(arg_list)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
