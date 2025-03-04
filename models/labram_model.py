from .abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .LaBraM.make_dataset import make_dataset, make_dataset_abnormal
import argparse
from pathlib import Path
from .LaBraM import utils
import torch
from timm.models import create_model
import torch.backends.cudnn as cudnn
from timm.utils import ModelEma
from timm.utils import NativeScaler
from collections import OrderedDict
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.data.distributed import DistributedSampler
import os
import random
import numpy as np
import time
import datetime
import json
from .LaBraM.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from .LaBraM.engine_for_finetuning import train_one_epoch, evaluate
from einops import rearrange
from mne.io import BaseRaw


def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int) # could be 50 or 30
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.65) # or 0.65

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/models/LaBraM/checkpoints/labram-base.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/models/LaBraM/checkpoints/finetune_tuab_base/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/log/finetune_tuab_base',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False) # TODO set to True

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


class LaBraMModel(AbstractModel):
    def __init__(
        self,
    ):
        super().__init__("LaBraMModel")
        assert torch.cuda.is_available(), "CUDA is not available"
        self.args, self.ds_init = get_args()
        if self.args.output_dir:
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        utils.init_distributed_mode(self.args)

        if self.ds_init is not None:
            utils.create_ds_config(self.args)

        #print(self.args)
        self.device = torch.device(self.args.device)

        # fix the seed for reproducibility
        seed = self.args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

        cudnn.benchmark = True

        self.model = create_model(
            self.args.model,
            pretrained=False,
            num_classes=self.args.nb_classes,
            drop_rate=self.args.drop,
            drop_path_rate=self.args.drop_path,
            attn_drop_rate=self.args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=self.args.use_mean_pooling,
            init_scale=self.args.init_scale,
            use_rel_pos_bias=self.args.rel_pos_bias,
            use_abs_pos_emb=self.args.abs_pos_emb,
            init_values=self.args.layer_scale_init_value,
            qkv_bias=self.args.qkv_bias,
        )

        patch_size = self.model.patch_size
        print("Patch size = %s" % str(patch_size))
        self.args.window_size = (1, self.args.input_size // patch_size)
        self.args.patch_size = patch_size

        if self.args.finetune:
            if self.args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.args.finetune, map_location='cpu')

            print("Load ckpt from %s" % self.args.finetune)
            checkpoint_model = None
            for model_key in self.args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None) and (self.args.model_filter_name != ''):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = self.model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            utils.load_state_dict(self.model, checkpoint_model, prefix=self.args.model_prefix)

        self.model.to(self.device)

        self.model_ema = None
        if self.args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(
                self.model,
                decay=self.args.model_ema_decay,
                device='cpu' if self.args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % self.args.model_ema_decay)

        self.model_without_ddp = self.model
        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        #print("Model = %s" % str(self.model_without_ddp))
        print('number of params:', self.n_parameters)


    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        if isinstance(X[0], np.ndarray):
            datasets_train_list = [make_dataset(X_, y_, meta_["sampling_frequency"], meta_["channel_names"]) for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta)]
        elif isinstance(X[0][0], BaseRaw):
            datasets_train_list = [make_dataset_abnormal(X_, y_) for X_, y_, meta_ in zip(cast(List[List[BaseRaw]], X), cast(List[List[str]], y), meta)]
        else:
            print(type(X[0][0]))
            raise ValueError("X must be a list of numpy arrays or a list of BaseRaw objects")
        datasets_train_list = [dataset for dataset in datasets_train_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in datasets_train_list]
        self.args.nb_classes = 2
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        dataset_test_list, dataset_val_list = None, None

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum([len(dataset) for dataset in datasets_train_list]) // self.args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in datasets_train_list:
            sampler_train = DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))
        sampler_test_list = []
        if dataset_test_list is not None:
            for dataset in dataset_test_list: # type: ignore
                sampler_test = DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
                )
                sampler_test_list.append(sampler_test)
                print("Sampler_test = %s" % str(sampler_test))
        sampler_val_list = []
        if dataset_test_list is not None:
            for dataset in dataset_val_list: # type: ignore
                sampler_val = DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
                )
                sampler_val_list.append(sampler_val)
                print("Sampler_val = %s" % str(sampler_val))


        if global_rank == 0 and self.args.log_dir is not None:
            os.makedirs(self.args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=self.args.log_dir)
        else:
            log_writer = None

        data_loader_train_list = [DataLoader(
                dataset, sampler=sampler,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=True,
            ) for dataset, sampler in zip(datasets_train_list, sampler_train_list)]
        if dataset_val_list is not None:
            data_loader_val_list = [DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * self.args.batch_size),
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_val_list, sampler_val_list)]
        else:
            data_loader_val_list = None
        if dataset_test_list is not None:
            data_loader_test_list = [DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * self.args.batch_size),
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test_list, sampler_test_list)]
        else:
            data_loader_test_list = None
        
        total_batch_size = self.args.batch_size * self.args.update_freq * utils.get_world_size()
        number_of_training_examples = sum([len(dataset) for dataset in datasets_train_list])
        num_training_steps_per_epoch = number_of_training_examples // total_batch_size
        print("LR = %.8f" % self.args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % self.args.update_freq)
        print("Number of training examples = %d" % number_of_training_examples)
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = self.model_without_ddp.get_num_layers()
        if self.args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(self.args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = self.model.no_weight_decay() # type: ignore
        if self.args.disable_weight_decay_on_rel_pos_bias:
            for i in range(num_layers):
                skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)    
    
        optimizer = create_optimizer(
            self.args, self.model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler() #None # #GradScaler() #

        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            self.args.lr, self.args.min_lr, self.args.epochs, num_training_steps_per_epoch,
            warmup_epochs=self.args.warmup_epochs, warmup_steps=self.args.warmup_steps,
        )
        if self.args.weight_decay_end is None:
            self.args.weight_decay_end = self.args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            self.args.weight_decay, self.args.weight_decay_end, self.args.epochs, num_training_steps_per_epoch)
        print("wd_schedule_values = %s" % str(wd_schedule_values))
        if len(wd_schedule_values) > 0:
            print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        if self.args.nb_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif self.args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))

        utils.auto_load_model(
            args=self.args, model=self.model, model_without_ddp=self.model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=self.model_ema)
                
        if self.args.eval and data_loader_test_list is not None:
            balanced_accuracy = []
            accuracy = []
            for data_loader in data_loader_test_list:
                test_stats = evaluate(data_loader, self.model, self.device, header='Test:', ch_names=ch_names_list, metrics=metrics, is_binary=(self.args.nb_classes == 1))
                accuracy.append(test_stats['accuracy'])
                balanced_accuracy.append(test_stats['balanced_accuracy'])
            print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
            exit(0)

        print(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        max_accuracy_test = 0.0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                for data_loader_train in data_loader_train_list:
                    data_loader_train.sampler.set_epoch(epoch) # type: ignore
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * self.args.update_freq)

            train_stats = train_one_epoch(
                self.model, criterion, data_loader_train_list, optimizer,
                self.device, epoch, loss_scaler, max_norm=self.args.clip_grad, log_writer=log_writer, 
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=self.args.update_freq, 
                ch_names_list=ch_names_list, model_ema=self.model_ema, is_binary=self.args.nb_classes == 1
            )
            
            if self.args.output_dir and self.args.save_ckpt:
                utils.save_model(
                    args=self.args, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=self.model_ema, save_ckpt_freq=self.args.save_ckpt_freq)
                
            if data_loader_val_list is not None and data_loader_test_list is not None and dataset_val_list is not None and dataset_test_list is not None:
                for data_loader_val, data_loader_test in zip(data_loader_val_list,data_loader_test_list):
                    val_stats = evaluate(data_loader_val, self.model, self.device, header='Val:', ch_names=ch_names_list, metrics=metrics, is_binary=self.args.nb_classes == 1)
                    print(f"Accuracy of the network on the {len(dataset_val_list)} val EEG: {val_stats['accuracy']:.2f}%")
                    test_stats = evaluate(data_loader_test, self.model, self.device, header='Test:', ch_names=ch_names_list, metrics=metrics, is_binary=self.args.nb_classes == 1)
                    print(f"Accuracy of the network on the {len(dataset_test_list)} test EEG: {test_stats['accuracy']:.2f}%")
                
                    if max_accuracy < val_stats["accuracy"]:
                        max_accuracy = val_stats["accuracy"]
                        if self.args.output_dir and self.args.save_ckpt:
                            utils.save_model(
                                args=self.args, model=self.model, model_without_ddp=self.model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch="best", model_ema=self.model_ema)
                        max_accuracy_test = test_stats["accuracy"]

                    print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
                    if log_writer is not None:
                        for key, value in val_stats.items():
                            if key == 'accuracy':
                                log_writer.update(accuracy=value, head="val", step=epoch)
                            elif key == 'balanced_accuracy':
                                log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                            elif key == 'f1_weighted':
                                log_writer.update(f1_weighted=value, head="val", step=epoch)
                            elif key == 'pr_auc':
                                log_writer.update(pr_auc=value, head="val", step=epoch)
                            elif key == 'roc_auc':
                                log_writer.update(roc_auc=value, head="val", step=epoch)
                            elif key == 'cohen_kappa':
                                log_writer.update(cohen_kappa=value, head="val", step=epoch)
                            elif key == 'loss':
                                log_writer.update(loss=value, head="val", step=epoch)
                        for key, value in test_stats.items():
                            if key == 'accuracy':
                                log_writer.update(accuracy=value, head="test", step=epoch)
                            elif key == 'balanced_accuracy':
                                log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                            elif key == 'f1_weighted':
                                log_writer.update(f1_weighted=value, head="test", step=epoch)
                            elif key == 'pr_auc':
                                log_writer.update(pr_auc=value, head="test", step=epoch)
                            elif key == 'roc_auc':
                                log_writer.update(roc_auc=value, head="test", step=epoch)
                            elif key == 'cohen_kappa':
                                log_writer.update(cohen_kappa=value, head="test", step=epoch)
                            elif key == 'loss':
                                log_writer.update(loss=value, head="test", step=epoch)
                    
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'n_parameters': self.n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': self.n_parameters}

            if self.args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        if isinstance(X[0], np.ndarray):
            datasets_test_list = [make_dataset(X_, None, meta_["sampling_frequency"], meta_["channel_names"]) for X_, meta_ in zip(cast(List[np.ndarray], X), meta)]
        elif isinstance(X[0][0], BaseRaw):
            datasets_test_list = [make_dataset_abnormal(X_, None) for X_, meta_ in zip(cast(List[List[BaseRaw]], X), meta)]
        else:
            print(type(X[0][0]))
            raise ValueError("X must be a list of numpy arrays or a list of BaseRaw objects")
        datasets_test_list = [dataset for dataset in datasets_test_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in datasets_test_list]
        self.args.nb_classes = 2

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train_list = []
        for dataset in datasets_test_list:
            sampler_train = DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))

        data_loader_test_list = [DataLoader(
                dataset, sampler=sampler,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=True,
            ) for dataset, sampler in zip(datasets_test_list, sampler_train_list)]
            
        self.model.eval()
        self.model.zero_grad()

        pred = []
        for data_loader, ch_names in zip(data_loader_test_list, ch_names_list):
            for step, batch in enumerate(data_loader):
                input_chans = None
                if ch_names is not None:
                    input_chans = utils.get_input_chans(ch_names)
                else:
                    print("error: ch_names is None")
                batch = batch.float().to(self.device, non_blocking=True) / 100
                batch = rearrange(batch, 'B N (A T) -> B N A T', T=200)

                # compute output
                with torch.cuda.amp.autocast(): # type: ignore
                    with torch.no_grad(): 
                        output = self.model(batch, input_chans)
                output = output.cpu().detach().numpy()
                output = np.argmax(output, axis=-1)

                pred.append(output)
            #print("pred = %s" % str(pred))
        # Inverse the mapping
        inverse_mapping = {0: 'left_hand', 1: 'right_hand'}
        pred = [inverse_mapping[p] for p in np.array(pred).flatten()]
        print(np.array(pred).flatten())
        print(len(np.array(pred).flatten()))
        return np.array(pred).flatten()
