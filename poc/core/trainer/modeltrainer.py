"""Implement the .train function."""

import torch
torch.backends.cudnn.benchmark = True
import numpy as np
from collections import defaultdict
from .scheduler import GradualWarmupScheduler

NON_BLOCKING = False


def train_model(model, loss_fn, trainloader, validloader, args, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, args)

    for epoch in range(args.trained_epochs):
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, args, setup, stats)

        if epoch == (args.trained_epochs - 1):
            model.eval()
            validate_model(model, loss_fn, validloader, args, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)

        if args.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break

    return stats

def step(model, loss_fn, dataloader, optimizer, scheduler, args, setup, stats):
    """Step through one epoch."""
    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

        # Get loss
        outputs = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if args.trained_scheduler == 'cyclic':
            scheduler.step()
        if args.dryrun:
            break
    if args.trained_scheduler == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))


def validate_model(model, loss_fn, dataloader, args, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, epoch_metric = 0, 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(dataloader):
            # Transfer to GPU
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

            # Get loss and metric
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            metric, name, _ = loss_fn.metric(outputs, targets)

            epoch_loss += loss.item()
            epoch_metric += metric.item()

            if args.dryrun:
                break

    stats['valid_losses'].append(epoch_loss / (batch + 1))
    stats['valid_' + name].append(epoch_metric / (batch + 1))

def set_optimizer(model, args):
    """Build model optimizer and scheduler.

    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    if args.trained_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.trained_lr, momentum=0.9,
                                    weight_decay=args.trained_decay, nesterov=True)
    elif args.trained_optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.trained_lr, weight_decay=args.trained_decay)

    if args.trained_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[120 // 2.667, 120 // 1.6,
                                                                     120 // 1.142], gamma=0.1)
        # Scheduler is fixed to 120 epochs so that calls with fewer epochs are equal in lr drops.

    if args.trained_warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler


def print_status(epoch, loss_fn, optimizer, stats):
    """Print basic console printout every validation epoch."""
    current_lr = optimizer.param_groups[0]['lr']
    name, format = loss_fn.metric()
    print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
          f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{format}} | '
          f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{format}} |')
