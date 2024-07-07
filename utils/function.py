import importlib
import os
import pathlib

import torch
import torch.optim
from torch.backends import cudnn

import data
import model.models as models
from .builder import get_builder


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    assert isinstance(args.gpu, int) and 0 <= args.gpu <= 7, f"gpu_id={args.gpu} is not supported!"

    if hasattr(args, "distributed") and args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # ourselves based on the total number of GPUs of the current node.
            args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              output_device=args.gpu,
                                                              find_unused_parameters=False)
        else:
            raise NotImplementedError
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError
    if args.set.lower() in ['cifar10', 'cifar100']:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    return model


def get_dataset(args, logger):
    logger.info(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    # if args.set.lower() == "imagenet":
    #     assert args.label_smoothing == 0.1
    return dataset


def get_model(args, logger):
    logger.info("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](builder=get_builder(args=args, logger=logger), num_classes=args.num_classes)

    return model


def get_trainer(args, logger):
    logger.info(f"=> Using trainer from trainers.{args.trainers}")
    trainer = importlib.import_module(f"trainers.{args.trainers}")

    return trainer.train, trainer.validate


def pretrained(args, model, logger):
    if os.path.isfile(args.pretrained):
        logger.info("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(args.pretrained, map_location="cpu")["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                logger.info("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        logger.info("=> no pretrained weights found at '{}'".format(args.pretrained))


def resume(args, model, optimizer, logger):
    if os.path.isfile(args.resume):
        logger.info(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location="cpu")
        if args.start_epoch is None:
            logger.info(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        logger.info(f"=> No checkpoint found at '{args.resume}'")


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(f"runs/{config}/{args.name}")
    else:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{config}/{args.name}")

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / f"logs"
    ckpt_base_dir = run_base_dir / f"checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def get_optimizer(args, model, logger):
    for n, v in model.named_parameters():
        if v.requires_grad:
            logger.info(f"<DEBUG> gradient to {n}")

        if not v.requires_grad:
            logger.info(f"<DEBUG> no gradient to {n}")

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if len(v.shape) == 1 and v.requires_grad]
        rest_params = [v for n, v in parameters if len(v.shape) != 1 and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    return optimizer
