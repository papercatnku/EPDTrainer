from utils.miscellaneous import update_dict, load_config, stastics_detail
from utils.trainutils import load_resume, get_saved_model_path, push_cuda_data, get_cuda_data
from trainer.trainer_base import trainer_base
from easydict import EasyDict
from loguru import logger
import os
from collections import defaultdict

from tqdm import tqdm
import torch


class trainer_static(trainer_base):
    def __init__(self, args):
        super().__init__(args)
        return

    def train(self,):
        for self.cur_epoch in range(self.epoch_start, self.config.epochs):
            self.before_epoch()
            self.train_epoch()
            self.validate_epoch()
            self.after_epoch()
        self.train_finish()
        return

    def export(self,):
        return

    def netinf_in_adapt(self, data_batch):
        if isinstance(self.config.input_names, (list, tuple)):
            return [data_batch[k] for k in self.config.input_names]
        else:
            return data_batch[self.config.input_names]

    def netinf_out_adapt(self, netinf_res):
        out_dict = {}
        if isinstance(netinf_res, (list, tuple)):
            for res, k in zip(netinf_res, self.config.output_names):
                out_dict[k] = res
        else:
            out_dict[self.config.output_names[0]] = netinf_res
        return out_dict

    def train_epoch(self):
        epoch_avg_loss = defaultdict(lambda: 0)
        num_batches = len(self.train_dl)
        num_items = len(self.train_dl.dataset)
        num_log_step = max(num_batches // self.config.sw.num_log_per_epoch,
                           10) if self.config.sw.num_log_per_epoch else 0
        for i, batch_data_dict in tqdm(enumerate(self.train_dl)):
            if self.config.use_cuda:
                batch_data_dict = push_cuda_data(batch_data_dict)
            inf_data = self.netinf_in_adapt(batch_data_dict)
            pred = self.model(
                *inf_data)
            pred_dict = self.netinf_out_adapt(pred)

            loss_dict = self.loss(
                pred_dict,
                batch_data_dict
            )
            loss_2backward = 0
            for k, v in loss_dict.items():
                loss_2backward += v.mean()

            for k, v in loss_dict.items():
                epoch_avg_loss[k] += v.mean().item()

            # backward
            self.optimizer.zero_grad()
            loss_2backward.backward()
            self.optimizer.step()

            # scheduler
            if self.scheduler and self.config.schedule_step_phase == 'batch':
                self.scheduler.step()
            # evaluator
            if (self.train_decoder and self.train_evaluator):
                decoded_res = self.train_decoder(pred_dict)
                self.train_evaluator.feed_data(
                    decoded_res, batch_data_dict)
            # sw
            batch_loss_dict = {}
            for k, v in loss_dict.items():
                batch_loss_dict['loss_' + k] = v.mean().item()
            batch_loss_dict['loss_total'] = sum(batch_loss_dict.values())
            self.sw.add_scalars(
                'loss_train', batch_loss_dict, global_step=self.cur_iter)

            if self.config.sw.show_graph:
                self.sw.add_graph(self.model.module, inf_data)
                self.config.sw.show_graph = False

            if (i != 0 and self.config.sw.num_log_per_epoch != 0 and i % num_log_step == 0):
                current = i * self.config.batch_size

                loss_detail_str = stastics_detail(loss_dict)

                logger.info(
                    f'train loss:{loss_2backward:>8.5f}, {loss_detail_str} \t[{current:>5d}/{num_items:>5d}]'
                )
                # decode & viz if available
                if (self.train_decoder and self.train_visualizer):
                    pass

            self.cur_iter += 1

        epoch_loss2log = defaultdict(lambda: 0)

        for k in epoch_avg_loss.keys():
            epoch_loss2log['train_loss_' +
                           k] = epoch_avg_loss[k] / float(num_batches)
        epoch_loss2log['train_loss_sum'] = sum(epoch_loss2log.values())

        logger.info(
            f'[epoch: {self.cur_epoch + 1}]\t{stastics_detail(epoch_loss2log)}]'
        )

        self.sw.add_scalars(
            'epoch_loss',
            epoch_loss2log,
            self.cur_epoch + 1
        )

        return

    def before_epoch(self):
        logger.info(f'-' * 10 + f' epoch: {self.cur_epoch+1} ' + '-' * 10)
        if self.config.sw.show_lr:
            lr = self.scheduler.get_last_lr(
            )[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            self.sw.add_scalar('learn_rate', lr, self.cur_epoch)

        # evaluator
        if self.train_evaluator:
            self.train_evaluator.reset()
        if self.validate_evaluator:
            self.validate_evaluator.reset()

        self.model.train()

        return

    def after_epoch(self):

        if self.scheduler and self.config.schedule_step_phase == 'epoch':
            self.scheduler.step()

        if self.train_evaluator:
            ststcs2log = self.train_evaluator.write_tblog(
                self.sw, 'stastics', self.cur_epoch+1)

            logger.info(
                f"[epoch: {self.cur_epoch + 1}]\ttrain stastics \t{stastics_detail(ststcs2log)}]")
            self.train_evaluator.reset()

        if self.validate_evaluator:
            ststcs2log = self.validate_evaluator.write_tblog(
                self.sw, 'stastics', self.cur_epoch+1)

            logger.info(
                f"[epoch: {self.cur_epoch + 1}]\tvalidate stastics \t{stastics_detail(ststcs2log)}]")
            self.validate_evaluator.reset()

        # save checkpoint
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(
            self.make_checkpoint(),
            get_saved_model_path(self.config.save_dir, self.cur_epoch+1)
        )

        return

    def validate_epoch(self,):
        self.model.eval()

        epoch_avg_loss = defaultdict(lambda: 0)
        num_batches = len(self.validate_dl)
        num_items = len(self.validate_dl.dataset)

        for i, batch_data_dict in tqdm(enumerate(self.validate_dl)):
            if self.config.use_cuda:
                batch_data_dict = push_cuda_data(batch_data_dict)
            inf_data = self.netinf_in_adapt(batch_data_dict)
            pred = self.model(
                *inf_data)
            pred_dict = self.netinf_out_adapt(pred)
            loss_dict = self.loss(
                pred_dict,
                batch_data_dict
            )
            for k, v in loss_dict.items():
                epoch_avg_loss[k] += v.mean().item()
            pass

            if (self.validate_decoder and self.validate_evaluator):
                decoded_res = self.validate_decoder(pred_dict)
                self.validate_evaluator.feed_data(
                    decoded_res, batch_data_dict)
                if (self.validate_visualizer):
                    pass

        epoch_loss2log = defaultdict(lambda: 0)

        for k in epoch_avg_loss.keys():
            epoch_loss2log['validate_loss_' +
                           k] = epoch_avg_loss[k] / float(num_batches)

        epoch_loss2log['validate_loss_sum'] = sum(epoch_loss2log.values())

        self.sw.add_scalars(
            'epoch_loss',
            epoch_loss2log,
            self.cur_epoch + 1
        )

        logger.info(
            f'[epoch: {self.cur_epoch + 1}]\t{stastics_detail(epoch_loss2log)}]'
        )

        return

    def train_finish(self):
        # save checkpoint
        torch.save(
            self.make_checkpoint(),
            os.path.join(self.config.save_dir,
                         f'{self.config.exp_name}_final_ckpt.pth')
        )
        return

    def make_checkpoint(self):
        return {
            'epoch': self.cur_epoch,
            'model_state': self.model.module.state_dict(),
            'optimizer': type(self.optimizer),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler': type(self.scheduler),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'iter': self.cur_iter,
        }
