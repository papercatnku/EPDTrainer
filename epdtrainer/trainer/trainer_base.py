import os
from loguru import logger
from easydict import EasyDict
from utils.miscellaneous import update_dict, load_config, setup_log
from utils.trainutils import load_resume, create_model, create_optimizer, create_scheduler, get_saved_model_path
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class trainer_base:
    def __init__(self, args):
        self.load_config(args)
        self.init_model()
        if self.config.schema == 'train':
            self.train_init()
        return

    def load_config(self, args):
        self.config = EasyDict(
            config_path=args.config,
            num_workers=args.num_workers,
            schema=args.schema,
            exp_name=None,
            resume_epoch=None,
            resume=None,
            # default optimizers
            optimizer='sgd',
            scheduler='onecycle',
            schedule_step_phase='epoch',  # [batch|epoch]
            learn_rate=3e-4,
            epochs=args.epochs,
            use_cuda=torch.cuda.device_count() > 0,  # CUDA_VISIBLE_DEVICES
            sw=EasyDict(
                num_log_per_epoch=5,
                show_lr=True,
                batch_step=0,
                batch_viz_step=0,
                show_graph=True
            )
        )
        self.funcs = EasyDict()
        mod = load_config(args.config)
        update_dict(self.config, getattr(mod, 'config'))
        update_dict(self.funcs, getattr(mod, 'funcs'))

        # overwrite config from command-line arguments
        args = vars(args)
        for k in [
            'exp_name',
            'resume_epoch',
            'resume',
            # training used
            'batch_size',
            'epochs',
            'eval_ratio'
            # quantitization training
            'qat',
            'qat_tda4',
        ]:
            v = args.get(k)
            if v is not None:
                logger.info(
                    f'[W] Overwrite config.{k} from {self.config.get(k)} to {v}')
                self.config[k] = v

        exp_root = os.path.join(self.config.get(
            'exp_record_dir', './exp_record'), self.config["exp_name"])

        self.config.save_dir = os.path.join(
            exp_root, 'ckpt')
        self.config.log_dir = os.path.join(exp_root, 'log')

        setup_log(self.config.log_dir)

        # load checkpoint
        self.pretrained = None
        self.checkpoint = None
        if self.config.resume_epoch is not None:
            self.config.resume = get_saved_model_path(
                self.config.save_dir, self.config.resume_epoch)
            self.checkpoint = load_resume(self.config.resume)
        elif self.config.resume is not None:
            checkpoint = load_resume(self.config.resume)
            self.pretrained = checkpoint['model_state']

        if self.checkpoint:
            self.config.sw.batch_step = self.checkpoint.get(
                'config', {}).get('sw', {}).get('batch_step', 0)
            self.config.sw.batch_viz_step = self.checkpoint.get(
                'config', {}).get('sw', {}).get('batch_viz_step', 0)

        logger.info(
            f'config: {json.dumps(self.config, indent=2)}')

        return

    def run(self):
        if (self.config.schema == 'train'):
            return self.train()
        elif(self.config.schema == 'export'):
            return self.export()
        else:
            logger.error(f"unsupported schema:{self.config.schema}")
            return

    def train(self,):
        logger.info("train procedure need to be implemented")
        return

    def export(self,):
        logger.info("export procedure need to be implemented")
        return

    def init_model(self,):
        # qat not supported
        if self.config.schema == 'train':
            self.device = torch.device('cuda')
            model = create_model(self.funcs, self.checkpoint)

            if self.pretrained:
                model.load_state_dict(self.pretrained)
                pass
            self.model = nn.DataParallel(model).to(self.device)
            self.optimizer = create_optimizer(
                self.model, self.checkpoint, self.config, self.funcs)

        elif self.config.schema == 'eval':
            self.model = create_model(self.funcs, self.checkpoint)
            if self.pretrained:
                self.model.load_state_dict(self.pretrained)

        else:
            logger.error(f"unsupported schema {self.config.schema}")

        return

    def train_init(self):
        os.makedirs(
            self.config.log_dir, exist_ok=True)
        self.sw = SummaryWriter(
            self.config.log_dir, comment=self.config.exp_name)

        # loss
        self.loss = self.funcs.create_losses(self.config)

        self.scheduler = create_scheduler(
            self.optimizer, self.checkpoint, self.config, self.funcs)

        self.train_evaluator, self.validate_evaluator = self.funcs.create_evaluator(
            self.config)
        # used to decode network outputs
        self.train_decoder, self.validate_decoder = self.funcs.create_decoder(
            self.config)
        # used to visualize decoded result
        self.train_visualizer, self.validate_visualizer = self.funcs.creater_visualizer(
            self.config)

        #
        self.train_dl, self.validate_dl = self.funcs.create_data_loader(
            self.config)

        # TODO: enable model ema
        if self.config.get('ema'):
            pass
        #
        self.cur_epoch = 0
        self.epoch_start = 0
        iter_start = 0
        if self.checkpoint:
            self.epoch_start = self.checkpoint.get('epoch', 0)
            iter_start = self.checkpoint.get('iter', 0)
        self.cur_iter = iter_start
        return
