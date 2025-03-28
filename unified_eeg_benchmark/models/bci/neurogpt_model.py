from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
import time
import datetime
import json
from mne.io import BaseRaw
from torch import manual_seed
from .NeuroGPT.src.batcher.downstream_dataset import MotorImageryDataset, BCIDataset
from .NeuroGPT.src.encoder.conformer_braindecode import EEGConformer
from .NeuroGPT.src.decoder.make_decoder import make_decoder
from .NeuroGPT.src.embedder.make import make_embedder
from .NeuroGPT.src.decoder.unembedder import make_unembedder
from .NeuroGPT.src.trainer.make import make_trainer
from .NeuroGPT.src.utils import cv_split_bci
from .NeuroGPT.src.trainer.base import Trainer
from .NeuroGPT.src.model import Model
from sklearn.model_selection import train_test_split


def get_config(args: argparse.Namespace=None) -> Dict:
    """
    Make config from command line arguments (as created by get_args()).
    Performs additional formating of args required for calling train().
    """

    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size =  2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (
            args.embedding_dim%64
         ) == 0, f'embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})' 
        args.num_attention_heads = args.embedding_dim//64

    if args.run_name == 'none':
        args.run_name = f'{args.architecture}'

        if args.architecture != 'LinearBaseline':
            
            if 'Pretrained' not in args.architecture:
                args.run_name += f'_lrs-{args.num_hidden_layers}'

                args.run_name += f'_hds-{args.num_attention_heads}'

            # args.run_name += f'_embd-{args.embedding_dim}'
            # args.run_name += f'_train-{args.training_style}'
            # args.run_name += f'_lr-{str(args.learning_rate).replace(".", "")[1:]}'
            # args.run_name += f'_bs-{args.per_device_training_batch_size}'
            # args.run_name += f'_drp-{str(args.dropout).replace(".", "")}'
            args.run_name += f'_ChunkLen-{args.chunk_len}'
            args.run_name += f'_NumChunks-{args.num_chunks}'
            args.run_name += f'_ovlp-{args.chunk_ovlp}'

        else:
            args.run_name += f'_train-{args.training_style}'

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H')}"

    if args.training_style == "decoding":
        args.run_name += '-' + str(args.fold_i)

    if args.smoke_test == "True":
        args.run_name = f'smoke-test_{args.run_name}'

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = args.wandb_mode if args.wandb_mode in {'online', 'offline'} and args.local_rank in {-1, 0} else "disabled"
    
    config = vars(args)

    for arg in config:
        
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        
        elif config[arg] == 'none':
            config[arg] = None

        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config

def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )

    # Data pipeline settings:
    parser.add_argument(
        '--train-data-path',
        metavar='DIR',
        default='../../tuh_tensors/',
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )

    parser.add_argument(
        '--dst-data-path',
        metavar='DIR',
        default="/itet-stor/jbuerki/home/unified_eeg_benchmark/unified_eeg_benchmark/models/bci/NeuroGPT/bci2a_eeg_npz/",
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )

    parser.add_argument(
        '--parcellation-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input data parcellation (default: 1024). '
             '! This is fixed for the current up-/downstream data.'
    )
    parser.add_argument(
        '--pretrained-model',
        metavar='DIR',
        type=str,
        default='/itet-stor/jbuerki/home/unified_eeg_benchmark/unified_eeg_benchmark/models/bci/NeuroGPT/pretrained_model/pytorch_model.bin',
        help='checkpoint used to initialize model weights '
             '(default: none)'
    )

    # Embedder settings:    
    parser.add_argument(
        '--embedding-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input embedding '
             '(default: 1024)'
    )
    parser.add_argument(
        '--num-hidden-layers-embedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of layers of linear embedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-embedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze embedder weights during training '
             '(default: False) '
    )

    # UnEmbedder settings:
    parser.add_argument(
        '--num-hidden-layers-unembedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of hidden layers for linear unembedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-unembedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze unembedder weights during training '
             '(default: False) '
    )


    # Decoder settings:
    parser.add_argument(
        '--architecture',
        metavar='STR',
        default='GPT',
        choices=(
            'GPT',
            'PretrainedGPT2'
        ),
        type=str,
        help='Model architecture used for sequence modeling / decoding. '
             '(default: GPT) '
    )
    parser.add_argument(
        '--num-hidden-layers',
        metavar='INT',
        default=6,
        type=int,
        help='number of hidden model layers in --architecture '
             '(default: 4). '
             '! Does not apply to LinearBaseline; '
             '! Same number of hidden layers is used for decoder / encoder '
             'parts of autoencoder (ie., default creates encoder and decoder '
             'with 4 hidden layers each)'
    )
    parser.add_argument(
        '--num-attention-heads',
        metavar='INT',
        default=-1,
        type=int,
        help='number of attention heads per transformer layer '
             '(default: embedding-dim // 64). '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--intermediate-dim-factor',
        metavar='INT',
        default=4,
        type=int,
        help='scales feed-forward transformer layer dimension relative to '
             'embedding-dim: intermediate-dim-factor * embedding-dim '
             '(default: 4)'
    )
    parser.add_argument(
        '--hidden-activation',
        metavar='STR',
        default='gelu_new',
        choices=(
            'gelu',
            'gelu_new',
            'relu',
            'silu'
        ),
        type=str,
        help='type of hidden activation of transformer layers '
             '(default: gelu_new); '
             'one of {"gelu", "gelu_new", "relu", "silu"}. '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--freeze-decoder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture '
             '(default: False) '
    )
    parser.add_argument(
        '--freeze-decoder-without-pooler-heads',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture, without pooler layer and '
             ' is-next-pred / decoding heads '
             '(default: False) '
    )


    # Trainer settings:
    parser.add_argument(
        '--resume-from',
        metavar='DIR',
        type=str,
        default='none',
        help='continue training from specified checkpoint '
             '(default: none)'
    )
    parser.add_argument(
        '--training-style',
        metavar='STR',
        default='decoding',
        choices=(
            'CSM',
            'CSM_causal',
            'decoding'
        ),
        type=str,
        help='training framework / style (default: CSM); '
             'one of CSM, decoding'
    )
    parser.add_argument(
        '--decoding-target',
        metavar='STR',
        default='none',
        type=str,
        help='key for decoding target variable in .tar-files in --data'
             '(default: none). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--num-decoding-classes',
        metavar='INT',
        default=2, # TODO here we need to change the default value
        type=int,
        help='number of decoding classes (ie., mental states) in --data '
             '(default: 0). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--training-steps',
        metavar='INT',
        default=5000,
        type=int,
        help='number of training steps to perform '
             '(default: 400000)'
    )
    parser.add_argument(
        '--validation-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of validation steps to perform at evaluation time '
             '(default: 1000)'
    )
    parser.add_argument(
        '--test-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of test steps to perform at test time'
             '(default: 2000). '
             '! Test evaluation only performed if test set created by '
             'setting --n-test-subjects-per-dataset != -1'
    )
    parser.add_argument(
        '--per-device-training-batch-size',
        metavar='INT',
        default=32,
        type=int,
        help='batch size during training per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--per-device-validation-batch-size',
        metavar='INT',
        default=32,
        type=int,
        help='batch size during evaluation per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--optim',
        metavar='STR',
        default='adamw_hf',
        type=str,
        help='optimizer to use for training '
             '(default: adamw_hf) -> adamw from HuggingFrace transformer library. '
             'For other options see Huggingface TrainerArgs.'
    )
    parser.add_argument(
        '--learning-rate',
        metavar='FLOAT',
        default=1e-4,
        type=float,
        help='maximum learning rate during training '
             '(default: 1e-4)'
    )
    parser.add_argument(
        '--warmup-ratio',
        metavar='FLOAT',
        default=0.01,
        type=float,
        help='warm-up steps for linear learning rate scheduler '
             'specified as fraction of --training-steps '
             '(default: 0.01)'
    )
    parser.add_argument(
        '--weight-decay',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='weight decay strength (indicating l2-regularisation strength) '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--adam-beta-1',
        metavar='FLOAT',
        default=0.9,
        type=float,
        help='adam beta 1 (default: 0.9)'
    )
    parser.add_argument(
        '--adam-beta-2',
        metavar='FLOAT',
        default=0.999,
        type=float,
        help='adam beta 2 (default: 0.999)'
    )
    parser.add_argument(
        '--adam-epsilon',
        metavar='FLOAT',
        default=1e-8,
        type=float,
        help='adam beta 2 (default: 1e-8)'
    )
    parser.add_argument(
        '--max-grad-norm',
        metavar='FLOAT',
        default=1.0,
        type=float,
        help='maximum gradient clipping norm (default: 1.0)'
    )
    parser.add_argument(
        '--lr-scheduler',
        metavar='STR',
        default='linear',
        choices=(
            'linear',
            'constant_with_warmup',
            'none'
        ),
        type=str,
        help='learning rate scheduler; '
             'one of {linear, constant_with_warmup, none} '
             '(default: linear)'
    )
    parser.add_argument(
        '--dropout',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='dropout ratio for hidden layers of embedder and decoder model parts '
             '(default: 0.1)'
    )
    
    # Logging settings:
    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        type=str,
        default='results/models/upstream',
        help='path where training is logged '
             '(default: results/models/upstream)'
    )
    parser.add_argument(
        '--log-every-n-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='frequence of logging in training steps '
             '(default: 10000)'
    )
    parser.add_argument(
        '--run-name',
        metavar='STR',
        type=str,
        default='dst',
        help='descriptor of the training run used for logging and wandb; '
             '! if set to "none", a unique identifier is automatically created'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='STR',
        choices=(
            'online',
            'offline',
            'disabled'
        ),
        default='disabled',
        help='track training w/ wandb online or offline or not at all '
             '(default: disabled) '
             '! requires setting up weights-and-bias for this machine; '
             'see: https://docs.wandb.ai/'
    )
    parser.add_argument(
        '--wandb-project-name',
        metavar='STR',
        type=str,
        default='learning-from-brains',
        help='name of wandb project where data is logged '
             '(default: learning-from-brains)'
    )

    # Other settings:
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )
    parser.add_argument(
        '--set-seed',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        type=str,
        help='whether or not to set random seed (default: True)'
    )
    parser.add_argument(
        '--fp16',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to use 16-bit precision GPU training '
             '(default: True)'
    )
    parser.add_argument(
        '--deepspeed',
        metavar='DIR',
        default="none",
        type=str,
        help='location of deepspeed configuration file; '
             'automatically adds deepspeed functionality to training if specified '
             '(default: none)'
    )
    parser.add_argument(
        '--local_rank',
        metavar='INT',
        default=-1,
        type=int,
        help='Rank of the process during distributed training '
             '(default: -1)'
    )
    parser.add_argument(
        '--num-workers',
        metavar='INT',
        default=8,
        type=int,
        help='number of data loading workers '
             '(default: 0 -> load in main process)'
    )
    parser.add_argument(
        '--plot-model-graph',
        metavar='BOOL',
        default="False",
        type=str,
        choices=('True', 'False'),
        help='whether or not to save an image of the model graph to log-dir '
             '(default: False)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='BOOL',
        default="False",
        type=str,
        choices=("True", "False"),
        help='whetehr or not to run training in smoke test-mode '
             '(default: False)'
             'If set to "True", training is restricted by setting: '
             '--per-device-training_batch_size 2 '
             '--per-device-validation_batch_size 2 '
             '--training-steps 2 '
             '--validation-steps 2 '
             '--test-steps 2 '
             '--log-every-n-steps 1'
    )
    parser.add_argument(
        '--bold-dummy-mode',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to replace BOLD with dummy during training; '
             'for internal testing purposes only! '
             '(default: False)'
    )
    parser.add_argument(
        '--do-train',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to run training '
             '(default: True). '
             'If "False", train() still returns trainer'
    )
    
    parser.add_argument(
        '--n-positions',
        metavar='INT',
        default=512,
        type=int,
        help='maximum sequence length that transformer model might ever be used with '
             '(default: 512)'
    )
    ## EEG settings
    parser.add_argument(
        '--chunk_len',
        default=500,
        type=int)
    parser.add_argument(
    '--num_chunks',
    default=2,
    type=int)
    parser.add_argument(
    '--chunk_ovlp',
    default=0,
    type=int)
    parser.add_argument(
    '--sampling_rate',
    default=250,
    type=int)
    parser.add_argument(
    '--fold_i',
    default=0,
    type=int)

    parser.add_argument(
        '--use-encoder',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether to use encoder or not'
    )
    parser.add_argument(
        '--do-normalization',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether to use encoder or not'
    )

    parser.add_argument('--filter-time-length', metavar='INT', default=25, type=int, help='length of the temporal filter (default: 25)')
    parser.add_argument('--pool-time-length', metavar='INT', default=75, type=int, help='length of temporal pooling filter (default: 75)')
    parser.add_argument('--stride-avg-pool', metavar='INT', default=15, type=int, help='length of stride between temporal pooling filters (default: 15)')
    parser.add_argument('--n-filters-time', metavar='INT', default=40, type=int, help='number of temporal filters (default: 40)')
    parser.add_argument('--num-encoder-layers', metavar='INT', default=6, type=int, help='number of transformer layers in encoder')

    parser.add_argument('--eval_every_n_steps', default=500, type=int)
    parser.add_argument('--freeze-encoder', metavar='BOOL', default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze encoder weights during training '
             '(default: False) '
    )
    parser.add_argument('--ft-only-encoder', 
                        metavar='BOOL', 
                        default='True',
        choices=('True', 'False'),
        type=str,
        help='finetune with only encoder or not '
             '(default: False) '
    )

    return parser

def make_model(model_config: Dict=None):
    """Make model from model_config 
    (as generated by get_config()).
    """
    if model_config["use_encoder"] == True:
        chann_coords = None
        
        encoder = EEGConformer(n_outputs=model_config["num_decoding_classes"], n_chans=22, n_times=model_config['chunk_len'], ch_pos=chann_coords, is_decoding_mode=model_config["ft_only_encoder"])
        #calculates the output dimension of the encoder, which is the output of transformer layer.
        model_config["parcellation_dim"] = ((model_config['chunk_len'] - model_config['filter_time_length'] + 1 - model_config['pool_time_length']) // model_config['stride_avg_pool'] + 1) * model_config['n_filters_time']

    else:
        encoder = None
        model_config["parcellation_dim"] = model_config["chunk_len"] * 22

    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"], # flattened, channel x chunk length
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        n_positions=model_config["n_positions"]
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"]
    )

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )
    else:
        print("No Embedder and Unembedder!")
        unembedder = None

    model = Model(
        encoder=encoder,
        embedder=embedder,
        decoder=decoder,
        unembedder=unembedder
    )

    if model_config["ft_only_encoder"]:
        model.switch_ft_mode(ft_encoder_only=True)

    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=model_config["num_decoding_classes"]
        )

    if model_config["pretrained_model"] is not None:
        model.from_pretrained(model_config["pretrained_model"])

    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters():
            param.requires_grad = False

    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters():
            param.requires_grad = False

    if model_config["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            if 'fc.' in name \
            or 'final_layer' in name:
                continue
            else:
                param.requires_grad = False

    if 'freeze_decoder_without_pooler_heads' in model_config \
        and model_config["freeze_decoder_without_pooler_heads"]:
        for name, param in model.decoder.named_parameters():
            if 'pooler_layer' in name \
            or 'decoding_head' in name \
            or 'is_next_head' in name:
                continue
            else:
                param.requires_grad = False

    if model_config["freeze_unembedder"] and unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    return model


class NeuroGPTModel(AbstractModel):
    def __init__(
        self,
    ):
        super().__init__("NeuroGPTModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.config = get_config()

        if self.config['do_train']:
            os.makedirs(
                self.config["log_dir"],
                exist_ok=True
            )
            resume_path = str(self.config["resume_from"]) if self.config["resume_from"] is not None else None
            
            if resume_path is not None:
                config_filepath = os.path.join(
                    self.config["resume_from"],
                    'train_config.json'
                )

                if os.path.isfile(config_filepath):
                    print(
                        f'Loading training config from {config_filepath}'
                    )

                    with open(config_filepath, 'r') as f:
                        self.config = json.load(f)

                else:

                    with open(config_filepath, 'w') as f:
                        json.dump(self.config, f, indent=2)
                
                checkpoints = [
                    int(p.split('checkpoint-')[1])
                    for p in os.listdir(resume_path)
                    if 'checkpoint-' in p
                    and os.path.isdir(os.path.join(resume_path, p))
                ]
                last_checkpoint = max(checkpoints)
                print(
                    f'Resuming training from checkpoint-{last_checkpoint} in {resume_path}'
                )
                self.config["resume_from"] = os.path.join(
                    resume_path,
                    f'checkpoint-{last_checkpoint}'
                )

            else:
                config_filepath = os.path.join(
                    self.config["log_dir"],
                    'train_config.json'
                )
                
                with open(config_filepath, 'w') as f:
                    json.dump(self.config, f, indent=2)

                self.config["resume_from"] = None

        assert self.config["training_style"] in {
            'CSM',
            'CSM_causal',
            'decoding'
        }, f'{self.config["training_style"]} is not supported.'
        
        assert self.config["architecture"] in {
            'GPT',
            'PretrainedGPT2'
        }, f'{self.config["architecture"]} is not supported.'
        
        if self.config['set_seed']:
            random.seed(self.config["seed"])
            manual_seed(self.config["seed"])

    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        if self.config["training_style"] == 'decoding':
            downstream_path = self.config["dst_data_path"]
        
            train_folds, test_folds = cv_split_bci(sorted(os.listdir(downstream_path))[:18])
            train_files = train_folds[self.config['fold_i']]
            test_files = test_folds[self.config['fold_i']]

            #self.config["num_decoding_classes"] = 4

            """
            train_dataset = MotorImageryDataset(train_files, sample_keys=[
                    'inputs',
                    'attention_mask'
                ], chunk_len=self.config["chunk_len"], num_chunks=self.config["num_chunks"], ovlp=self.config["chunk_ovlp"], root_path=downstream_path, gpt_only= not self.config["use_encoder"])
            # pdb.set_trace()
            
            self.test_dataset = MotorImageryDataset(test_files, sample_keys=[
                    'inputs',
                    'attention_mask'
                ], chunk_len=self.config["chunk_len"], num_chunks=self.config["num_chunks"], ovlp=self.config["chunk_ovlp"], root_path=downstream_path, gpt_only= not self.config["use_encoder"])
            """
            task = meta[0]["task_name"]

            train_dataset = BCIDataset(X, y, meta, task, sample_keys=[
                    'inputs',
                    'attention_mask'
                ], chunk_len=self.config["chunk_len"], num_chunks=self.config["num_chunks"], ovlp=self.config["chunk_ovlp"], root_path="", gpt_only= not self.config["use_encoder"])
            validation_dataset = MotorImageryDataset(task, test_files, sample_keys=[
                    'inputs',
                    'attention_mask'
                ], chunk_len=self.config["chunk_len"], num_chunks=self.config["num_chunks"], ovlp=self.config["chunk_ovlp"], root_path=downstream_path, gpt_only= not self.config["use_encoder"])
        else:
            print("ERROR: Invalid training style")

        def model_init(params: Dict=None):
            model_config = dict(self.config)
            if params is not None:
                model_config |= params

            return make_model(model_config)

        if self.config["training_style"] == "decoding":
            model_save_steps = self.config["training_steps"]*2
        else:
            model_save_steps = self.config["log_every_n_steps"]

        self.trainer = make_trainer(
            model_init=model_init,
            training_style=self.config["training_style"],
            run_name=self.config["run_name"],
            output_dir=self.config["log_dir"],
            train_dataset=train_dataset,
            validation_dataset=train_dataset,
            per_device_train_batch_size=self.config["per_device_training_batch_size"],
            per_device_eval_batch_size=self.config["per_device_validation_batch_size"],
            dataloader_num_workers=self.config["num_workers"],
            optim=self.config["optim"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            adam_beta1=self.config["adam_beta_1"],
            adam_beta2=self.config["adam_beta_1"],
            adam_epsilon=self.config["adam_epsilon"],
            max_grad_norm=self.config["max_grad_norm"],
            lr_scheduler_type=self.config["lr_scheduler"],
            warmup_ratio=self.config["warmup_ratio"],
            max_steps=self.config["training_steps"],
            # num_train_epochs=5,
            save_steps=model_save_steps,
            logging_steps=self.config["log_every_n_steps"],
            eval_steps=self.config["eval_every_n_steps"],
            seed=self.config["seed"] if self.config['set_seed'] else np.random.choice(range(1, 100000)),
            fp16=self.config["fp16"],
            deepspeed=self.config["deepspeed"],
        )

        if self.config['do_train']:
            self.trainer.train(resume_from_checkpoint=self.config["resume_from"])
            self.trainer.save_model(
                os.path.join(
                    self.config["log_dir"],
                    'model_final'
                )
            )

    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")

        task = meta[0]["task_name"]

        self.test_dataset = BCIDataset(X, None, meta, task, sample_keys=[
                    'inputs',
                    'attention_mask'
                ], chunk_len=self.config["chunk_len"], num_chunks=self.config["num_chunks"], ovlp=self.config["chunk_ovlp"], root_path="", gpt_only= not self.config["use_encoder"])

        if self.test_dataset is not None:
            test_prediction = self.trainer.predict(self.test_dataset)
            #print(test_prediction)
            pd.DataFrame(
                test_prediction.metrics,
                index=[0]
            ).to_csv(
                os.path.join(
                    self.config["log_dir"],
                    'test_metrics.csv'
                ),
                index=False
            )
            np.save(
                os.path.join(
                    self.config["log_dir"],
                    'test_predictions.npy'
                ),
                test_prediction.predictions
            )
            np.save(
                os.path.join(
                    self.config["log_dir"],
                    'test_label_ids.npy'
                ),
                test_prediction.label_ids
            )

        if test_prediction.label_ids is None:
            def softmax(logits):
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # for numerical stability
                return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Convert logits to probabilities
            probabilities = softmax(test_prediction.predictions["outputs"])

            # Get the predicted classes (0 or 1) by taking argmax along axis 1 (the class axis)
            pred = np.argmax(probabilities, axis=1)
        else:
            pred = test_prediction.label_ids

        pred = self.test_dataset.decode_predictions(pred)
        #print(pred)
        #print(pred.shape)
        return pred
        
