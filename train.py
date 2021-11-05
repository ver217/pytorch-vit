import argparse
import math
import os
import random
import shutil
from datetime import datetime

from torch.serialization import load

from config import configs

import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import ImageNetFolder, make_meters, DaliImageNet
from optim import lr_scheduler
from timm.models import vit_small_patch16_224
from torch.cuda import amp

METRIC = 'acc/test_top1'


def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_batches_per_step', type=int, default=1)
    parser.add_argument('--dataset_path')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--warmup_epochs', type=float)
    parser.add_argument('--log_name', default='vit')
    parser.add_argument('--tensorboard_dir', default='./tb_logs')
    parser.add_argument('--ckpt_dir', default='./ckpt')
    parser.add_argument('--use_tensorboard',
                        default=None, action='store_true')
    parser.add_argument('--save_checkpoint',
                        default=None, action='store_true')
    parser.add_argument('--dali',
                        default=None, action='store_true')
    parser.add_argument('--gpu_aug',
                        default=None, action='store_true')
    parser.add_argument('--amp',
                        default=None, action='store_true')
    args = parser.parse_args()

    ##################
    # Update configs #
    ##################
    for k, v in configs.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in vars(args).items():
        printr(f'[{k}] = {v}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.set_num_threads(args.num_threads)

    save_path = os.path.join(args.ckpt_dir, args.log_name)
    printr(f'[save_path] = {save_path}')
    checkpoint_path = os.path.join(save_path, 'checkpoints')
    checkpoint_path_fmt = os.path.join(
        checkpoint_path, f'e{"{epoch}"}-r{dist.get_rank()}.pth'
    )
    latest_pth_path = os.path.join(
        checkpoint_path, f'latest-r{dist.get_rank()}.pth'
    )
    best_pth_path = os.path.join(
        checkpoint_path, f'best-r{dist.get_rank()}.pth'
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.evaluate:
        latest_pth_path = best_pth_path

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    printr(f'\n==> creating dataset from "{args.dataset_path}"')
    if args.dali:
        dataset = DaliImageNet(args.dataset_path,
                               batch_size=args.batch_size,
                               shard_id=dist.get_rank(),
                               num_shards=dist.get_world_size(),
                               gpu_aug=args.gpu_aug)
    else:
        dataset = ImageNetFolder(args.dataset_path)
        # Horovod: limit # of CPU threads to be used per worker.
        loader_kwargs = {'num_workers': args.num_workers,
                         'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers
        # instead of 'fork' to prevent issues with Infiniband implementations
        # that are not fork-safe
        if (loader_kwargs.get('num_workers', 0) > 0 and
                hasattr(mp, '_supports_context') and
                mp._supports_context and
                'forkserver' in mp.get_all_start_methods()):
            loader_kwargs['multiprocessing_context'] = 'forkserver'
        printr(f'\n==> loading dataset "{loader_kwargs}""')

    if args.dali:
        samplers, loaders = {split: None for split in dataset}, dataset
    else:
        samplers, loaders = {}, {}
        for split in dataset:
            samplers[split] = torch.utils.data.distributed.DistributedSampler(
                dataset[split], num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loaders[split] = torch.utils.data.DataLoader(
                dataset[split], batch_size=args.batch_size,
                sampler=samplers[split],
                drop_last=(args.num_batches_per_step > 1
                           and split == 'train'),
                **loader_kwargs
            )

    printr(f'\n==> creating model {vit_small_patch16_224}')
    model = vit_small_patch16_224().cuda()
    model = DDP(model, device_ids=[dist.get_rank()])

    criterion = nn.CrossEntropyLoss()
    # Horovod: scale learning rate by the number of GPUs.

    printr(f'\n==> creating optimizer Adam with LR = {args.lr}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # resume from checkpoint
    last_epoch, best_metric = -1, None
    if os.path.exists(latest_pth_path):
        printr(f'\n[resume_path] = {latest_pth_path}')
        checkpoint = torch.load(latest_pth_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get(
            f'{METRIC}_best', best_metric)
    else:
        printr('\n==> train from scratch')

    num_steps_per_epoch = len(loaders['train'])
    warmup_lr_epochs = getattr(args, 'warmup_epochs', 0)

    last = max((last_epoch - warmup_lr_epochs + 1)
               * num_steps_per_epoch - 2, -1)
    decay_steps = args.num_epochs * num_steps_per_epoch
    warmup_steps = warmup_lr_epochs
    if warmup_lr_epochs > 0:
        warmup_steps *= num_steps_per_epoch

    scheduler = lr_scheduler.CosineAnnealingWarmup(
        optimizer, decay_steps, warmup_steps, last_epoch=last)

    ############
    # Training #
    ############

    training_meters = make_meters()
    meters = evaluate(model, meters=training_meters,
                      loader=loaders['test'], split='test', dali=args.dali)
    for k, meter in meters.items():
        printr(f'[{k}] = {meter:.2f}')
    if args.evaluate or last_epoch >= args.num_epochs:
        return

    if dist.get_rank() == 0 and args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_path = os.path.join(args.tensorboard_dir, args.log_name)
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    for current_epoch in range(last_epoch + 1, args.num_epochs):
        printr(f'\n==> training epoch {current_epoch + 1}/{args.num_epochs}')

        train(model=model, loader=loaders['train'],
              epoch=current_epoch,
              sampler=samplers['train'], criterion=criterion,
              optimizer=optimizer, scheduler=scheduler,
              schedule_lr_per_epoch=False,
              writer=writer, show_progress=dist.get_rank() == 0, dali=args.dali)

        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader,
                                       meters=training_meters,
                                       split=split, show_progress=dist.get_rank() == 0, dali=args.dali))

        best = False
        if best_metric is None or best_metric < meters[METRIC]:
            best_metric, best = meters[METRIC], True
        meters[f'{METRIC}_best'] = best_metric

        printr('')
        for k, meter in meters.items():
            printr(f'[{k}] = {meter:.2f}')
            if writer is not None:
                writer.add_scalar(k, meter, current_epoch)

        checkpoint = {
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meters': meters
        }

        # save checkpoint
        if args.save_checkpoint:
            checkpoint_path = checkpoint_path_fmt.format(epoch=current_epoch)
            torch.save(checkpoint, checkpoint_path)
            shutil.copyfile(checkpoint_path, latest_pth_path)
            if best:
                shutil.copyfile(checkpoint_path, best_pth_path)
            if current_epoch >= 3:
                os.remove(
                    checkpoint_path_fmt.format(epoch=current_epoch - 3)
                )
            printr(f'[save_path] = {checkpoint_path}')


def train(model, loader, epoch, sampler, criterion, optimizer,
          scheduler, schedule_lr_per_epoch, writer=None, show_progress=True, dali=False):
    if sampler:
        sampler.set_epoch(epoch)
    model.train()
    for step, (inputs, targets) in enumerate(tqdm(
            loader, desc='train', ncols=0, disable=not show_progress)):
        if not dali:
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # write train loss log
        dist.all_reduce(loss)
        loss = (loss / dist.get_world_size()).item()
        if writer is not None:
            global_step = step + epoch * len(loader)
            writer.add_scalar('loss/train', loss, global_step)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr/train', lr, global_step)

        adjust_learning_rate(scheduler, epoch=epoch, step=step,
                             schedule_lr_per_epoch=schedule_lr_per_epoch)


def evaluate(model, loader, meters, split='test', show_progress=False, dali=False):
    _meters = {}
    for k, meter in meters.items():
        meter.reset()
        _meters[k.format(split)] = meter
    meters = _meters

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=split, ncols=0, disable=not show_progress):
            if not dali:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)

    buffer = torch.zeros(1, device=dist.get_rank())
    for k, meter in meters.items():
        data = meter.data()
        for dk, d in data.items():
            buffer.fill_(d)
            dist.all_reduce(buffer)
            data[dk] = buffer.item()
        meter.set(data)
        meters[k] = meter.compute()
    return meters


def adjust_learning_rate(scheduler, epoch, step,
                         schedule_lr_per_epoch=False):
    if schedule_lr_per_epoch and (step > 0 or epoch == 0):
        return
    else:
        scheduler.step()


def printr(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


if __name__ == '__main__':
    setup()
    main()
    cleanup()
