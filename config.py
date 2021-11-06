from utils import Config
from timm.models import vit_small_patch16_224
from optim import lr_scheduler
from torch import optim

__all__ = ['config']

config = Config({
    'train': {
        'seed': 42,
        'num_epochs': 300,
        'num_batches_per_step': 4,
        'log_name': 'vit-s16',
        'tensorboard_dir': '/mnt/shared/vit/tb_logs',
        'ckpt_dir': '/mnt/shared/vit/ckpt',
        'clip_grad': 1.0,
        'use_tensorboard': True,
        'save_checkpoint': False,
        'amp': True,
    },
    'data': {
        'dataset_path': '/mnt/shared/imagenet-100',
        'batch_size': 128,
        'dali': True,
        'gpu_aug': True,
        # 'num_workers': 4,
    },
    'optimizer': {
        'type': optim.AdamW,
        'lr': 3e-3,
        'weight_decay': 0.3,
    },
    'lr_scheduler': {
        'type': lr_scheduler.CosineAnnealingWarmup,
        'warmup_steps': 32,
    },
    'model': {
        'type': vit_small_patch16_224,
        'drop_rate': 0.1,
        'weight_init': 'jax',
    }
})
