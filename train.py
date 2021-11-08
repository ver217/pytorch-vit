import argparse
import os
import random
import shutil


from config import config

import numpy as np
import torch.distributed as dist
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import ImageNetFolder, make_meters, DaliImageNet
from torch.cuda import amp
from logger import DistributedLogger

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
    args = parser.parse_args()

    # Init logger
    logger = DistributedLogger(
        config.train.log_name, log_dir=config.train.log_dir)

    # Print config
    for k, v in config.items():
        logger.info(f'\n[{k}]')
        for name, val in v.items():
            logger.info(f'{name} = {val}')

    # Set seed
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Set ckpt path
    checkpoint_path = os.path.join(
        config.train.ckpt_dir, config.train.log_name)
    checkpoint_path_fmt = os.path.join(
        checkpoint_path, f'e{"{epoch}"}-r{dist.get_rank()}.pth'
    )
    latest_pth_path = os.path.join(
        checkpoint_path, f'latest-r{dist.get_rank()}.pth'
    )
    best_pth_path = os.path.join(
        checkpoint_path, f'best-r{dist.get_rank()}.pth'
    )
    if config.train.save_checkpoint:
        os.makedirs(checkpoint_path, exist_ok=True)

    if args.evaluate:
        latest_pth_path = best_pth_path

    # Build dataset
    logger.info(
        f'\nTotal batch size = {config.data.batch_size * dist.get_world_size() * config.train.num_batches_per_step}')
    if config.data.dali:
        dataset = DaliImageNet(config.data.dataset_path,
                               batch_size=config.data.batch_size,
                               shard_id=dist.get_rank(),
                               num_shards=dist.get_world_size(),
                               gpu_aug=config.data.gpu_aug)
    else:
        dataset = ImageNetFolder(config.data.dataset_path)
        loader_kwargs = {'num_workers': config.data.num_workers,
                         'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers
        # instead of 'fork' to prevent issues with Infiniband implementations
        # that are not fork-safe
        if (loader_kwargs.get('num_workers', 0) > 0 and
                hasattr(mp, '_supports_context') and
                mp._supports_context and
                'forkserver' in mp.get_all_start_methods()):
            loader_kwargs['multiprocessing_context'] = 'forkserver'

    if config.data.dali:
        samplers, loaders = {split: None for split in dataset}, dataset
    else:
        samplers, loaders = {}, {}
        for split in dataset:
            samplers[split] = torch.utils.data.distributed.DistributedSampler(
                dataset[split], num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loaders[split] = torch.utils.data.DataLoader(
                dataset[split], batch_size=config.data.batch_size,
                sampler=samplers[split],
                drop_last=(config.train.num_batches_per_step > 1
                           and split == 'train'),
                **loader_kwargs
            )

    # Build model
    model = config.model.pop('type')
    model = model(**config.model).cuda()
    model = DDP(model, device_ids=[dist.get_rank()])

    # Build loss function
    criterion = config.criterion.pop('type')
    criterion = criterion(**config.criterion)

    # Build optimizer
    optimizer = config.optimizer.pop('type')
    optimizer = optimizer(model.parameters(), **config.optimizer)

    scaler = None
    if config.train.amp:
        scaler = amp.GradScaler()

    # num_steps_per_epoch = len(loaders['train']) // args.num_batches_per_step
    # warmup_lr_epochs = getattr(args, 'warmup_epochs', 0)
    # warmup_lr_epochs = int(warmup_lr_epochs)
    # last = max((last_epoch - warmup_lr_epochs + 1)
    #            * num_steps_per_epoch - 2, -1)
    # decay_steps = args.num_epochs * num_steps_per_epoch
    # warmup_steps = warmup_lr_epochs
    # if warmup_lr_epochs > 0:
    #     warmup_steps *= num_steps_per_epoch
    # Build lr scheduler
    scheduler = config.lr_scheduler.pop('type')
    scheduler = scheduler(
        optimizer, config.train.num_epochs, **config.lr_scheduler)

    # Resume from checkpoint
    last_epoch, best_metric = -1, None
    if os.path.exists(latest_pth_path):
        logger.info(f'\n[resume_path] = {latest_pth_path}')
        checkpoint = torch.load(latest_pth_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get(
            f'{METRIC}_best', best_metric)
    else:
        logger.info('\n==> train from scratch')

    # Training
    training_meters = make_meters()
    meters = evaluate(model, meters=training_meters,
                      loader=loaders['test'], split='test', dali=config.data.dali)
    for k, meter in meters.items():
        logger.info(f'[{k}] = {meter:.2f}')
    if args.evaluate or last_epoch >= config.train.num_epochs:
        return

    if dist.get_rank() == 0 and config.train.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_path = os.path.join(
            config.train.tensorboard_dir, config.train.log_name)
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    for current_epoch in range(last_epoch + 1, config.train.num_epochs):
        logger.info(
            f'\n==> training epoch {current_epoch + 1}/{config.train.num_epochs}')

        train(model=model, loader=loaders['train'],
              epoch=current_epoch,
              sampler=samplers['train'], criterion=criterion,
              optimizer=optimizer,
              num_batches_per_step=config.train.num_batches_per_step,
              writer=writer, show_progress=dist.get_rank() == 0, dali=config.data.dali,
              use_amp=config.train.amp, scaler=scaler, clip_grad=config.train.clip_grad)
        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader,
                                       meters=training_meters,
                                       split=split, show_progress=dist.get_rank() == 0, dali=config.data.dali))

        best = False
        if best_metric is None or best_metric < meters[METRIC]:
            best_metric, best = meters[METRIC], True
        meters[f'{METRIC}_best'] = best_metric

        logger.info('')
        for k, meter in meters.items():
            logger.info(f'[{k}] = {meter:.2f}')
            if writer is not None:
                writer.add_scalar(k, meter, current_epoch)
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('lr/train', lr, current_epoch)

        scheduler.step()
        checkpoint = {
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meters': meters
        }

        # save checkpoint
        if config.train.save_checkpoint:
            checkpoint_path = checkpoint_path_fmt.format(epoch=current_epoch)
            torch.save(checkpoint, checkpoint_path)
            shutil.copyfile(checkpoint_path, latest_pth_path)
            if best:
                shutil.copyfile(checkpoint_path, best_pth_path)
            if current_epoch >= 3:
                os.remove(
                    checkpoint_path_fmt.format(epoch=current_epoch - 3)
                )
            logger.info(f'[save_path] = {checkpoint_path}')


def train(model,
          loader,
          epoch,
          sampler,
          criterion,
          optimizer,
          num_batches_per_step,
          writer=None,
          show_progress=True,
          dali=False,
          use_amp=False,
          scaler=None,
          clip_grad=0.0):
    if sampler:
        sampler.set_epoch(epoch)
    model.train()
    r_num_batches_per_step = 1.0 / num_batches_per_step
    # use drop last policy
    num_steps_per_epoch = len(loader) // num_batches_per_step
    train_iter = iter(loader)

    def run_step():
        inputs, targets = next(train_iter)
        if not dali:
            inputs = inputs.cuda()
            targets = targets.cuda()
        with amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.mul_(r_num_batches_per_step)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    with tqdm(range(num_steps_per_epoch), desc='train', ncols=0, disable=not show_progress) as t:
        for step in t:
            optimizer.zero_grad()
            loss = torch.zeros(1).cuda()
            for _ in range(num_batches_per_step - 1):
                with model.no_sync():
                    loss += run_step()
            loss += run_step()
            if clip_grad > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # write train loss log
            dist.all_reduce(loss)
            loss = (loss / dist.get_world_size()).item()
            t.set_postfix(loss=f'{loss:.4f}')
            if writer is not None:
                global_step = step + epoch * num_steps_per_epoch
                writer.add_scalar('loss/train', loss, global_step)
    try:
        while True:
            next(train_iter)
    except StopIteration:
        pass


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


if __name__ == '__main__':
    setup()
    main()
    cleanup()
