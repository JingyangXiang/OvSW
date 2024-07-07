import copy
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
from ruamel import yaml
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.tensorboard import SummaryWriter

from utils.function import (
    get_dataset,
    get_directories,
    get_model, get_optimizer,
    get_trainer,
    pretrained,
    resume,
    set_gpu
)
from utils.logger import create_logger, prRed
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import get_lr, get_parameters_num, init_model, save_checkpoint, update_epoch
from utils.schedulers import get_policy


def main():
    from args import args
    print(args)

    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    if hasattr(args, "distributed") and args.distributed:
        raise ValueError("distributed is not supported yet")
    else:
        main_worker(args)


def main_worker(args):
    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    if not log_base_dir.exists():
        os.makedirs(log_base_dir)
    logger = create_logger(output_dir=log_base_dir, name="{arch}".format(arch=args.arch))
    logger.info(args)
    train, validate = get_trainer(args, logger=logger)

    logger.info("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args, logger=logger)
    logger.info(model)

    init_model(model, logger, args)

    model = set_gpu(args, model)
    device_target = torch.cuda.get_device_name()
    logger.info(f"==> Run on {device_target}")

    get_parameters_num(model=model, logger=logger)

    if args.pretrained:
        pretrained(args, model, logger=logger)

    optimizer = get_optimizer(args, model, logger=logger)
    data = get_dataset(args, logger=logger)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer, logger=logger)

    # Data loading code
    if args.evaluate:
        validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    # save a yaml file to read to record parameters
    args_text = copy.copy(args.__dict__)
    if "ckpt_base_dir" in args_text.keys():
        del args_text['ckpt_base_dir']
    with open(run_base_dir / 'args.yaml', 'w', encoding="utf-8") as f:
        yaml.dump(args_text, f, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True, indent=4)

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = 0.
    acc5 = 0.

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    args.logger = logger
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        update_epoch(model, epoch=epoch)

        # lr changes when epoch < args.epochs
        lr_policy(epoch, iteration=None)
        cur_lr = get_lr(optimizer)
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0

        if is_best or save or epoch == args.epochs - 1:
            # if is_best:
            #     prRed(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}", logger=logger)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        prRed(f'=> Epoch: {epoch}, LR: {cur_lr:.4f}, Acc: {acc1:.2f}%, Best Acc: {best_acc1:.2f}%', logger)
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    device_target = torch.cuda.get_device_name()
    write_result_to_csv_scrach(
        arch=args.arch,
        set=args.set,
        base_config=args.config,
        name=args.name,
        conv_type=args.conv_type,
        bn_type=args.bn_type,
        nonlinearity=args.nonlinearity,
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        curr_acc1=acc1,
        curr_acc5=acc5,
        epochs=args.epochs,
        forward_type=args.forward_type,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_length=args.warmup_length,
        seed=args.seed,
        device_target=device_target,
        act_a=args.act_a,
        act_w=args.act_w,
        nesterov=args.nesterov,
        no_bn_decay=args.no_bn_decay,
        trainers=args.trainers,
        lr=args.lr,
        batch_size=args.batch_size,
        delta=args.delta,
        enable_dampen=args.enable_dampen,
        track_momentum=args.track_momentum,
        track_threshold=args.track_threshold,
        dampen_weight=args.dampen_weight,
        enable_ags=args.enable_ags
    )


def write_result_to_csv_scrach(**kwargs):
    results = pathlib.Path("runs") / f"train_{kwargs.get('arch')}_from_scrach.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Arch, "
            "Set, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Device Target, "
            "Epochs, "
            "Optimizer, "
            "Weight Decay, "
            "Warmup Length, "
            "Seed, "
            "Forward Type, "
            "Nesterov, "
            "NoBnDecay, "
            "Trainers, "
            "LearningRate, "
            "BatchSize, "
            "ConvType, "
            "BnType, "
            "Nonlinearity, "
            "ActActivation, "
            "ActWeight, "
            "Delta, "
            "Enable Dampening, "
            "Track Momentum, "
            "Track Threshold, "
            "Dampen Weight, "
            "Enable Ags\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            ("{now}, "
             "{base_config}, "
             "{name}, "
             "{arch}, "
             "{set}, "
             "{curr_acc1:.02f}, "
             "{curr_acc5:.02f}, "
             "{best_acc1:.02f}, "
             "{best_acc5:.02f}, "
             "{best_train_acc1:.02f}, "
             "{best_train_acc5:.02f}, "
             "{device_target}, "
             "{epochs}, "
             "{optimizer}, "
             "{weight_decay}, "
             "{warmup_length}, "
             "{seed}, "
             "{forward_type}, "
             "{nesterov}, "
             "{no_bn_decay}, "
             "{trainers}, "
             "{lr}, "
             "{batch_size}, "
             "{conv_type}, "
             "{bn_type}, "
             "{nonlinearity}, "
             "{act_a}, "
             "{act_w}, "
             "{delta}, "
             "{enable_dampen}, "
             "{track_momentum}, "
             "{track_threshold}, "
             "{dampen_weight}, "
             "{enable_ags}\n"
             ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
