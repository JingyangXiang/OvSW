import argparse
import ast
import sys

import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Reinforcement Quantization Training")

    # General Config
    parser.add_argument("--data", help="path to dataset base directory", default="/mnt/disk1/datasets")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, choices=["CIFAR10", "CIFAR100", "ImageNet"],
                        default="ImageNet")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture")
    parser.add_argument("--config", help="Config file to use (see configs dir)", default=None)
    parser.add_argument("--log-dir", help="Where to save the runs. If None use ./runs", default=None)
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=None, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current "
                             "node when using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate",
                        dest="lr")
    parser.add_argument("--warmup_length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("-p", "--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 100)")
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--resume", default="", type=str, metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("-e", "--evaluate", dest="evaluate", default=False, type=ast.literal_eval,
                        help="evaluate model on validation set")
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="gpu id for training")

    # Learning Rate Policy Specific
    parser.add_argument("--lr-policy", default="constant_lr", help="Policy for the learning rate.")
    parser.add_argument("--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr")
    parser.add_argument("--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier")
    parser.add_argument("--name", default=None, type=str, help="Experiment name to append to filepath")
    parser.add_argument("--save_every", default=-1, type=int, help="Save every ___ epochs")
    parser.add_argument("--nesterov", default=False, type=ast.literal_eval)
    parser.add_argument("--conv-type", type=str, default=None, help="What kind of sparsity to use")
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument("--nonlinearity", default="relu", help="Nonlinearity used by initialization")
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument("--init", default="kaiming_normal", help="Weight initialization modifications")
    parser.add_argument("--no-bn-decay", type=ast.literal_eval, default=False, help="No batchnorm decay")
    parser.add_argument("--label-smoothing", type=float, help="Label smoothing to use, default 0.0", default=None)
    parser.add_argument("--trainers", type=str, default="ovsw", help="cs, ss, or standard training")
    parser.add_argument("--score-init-constant", type=float, default=None, help="Sample Baseline Subnet Init")

    # Binary Network related
    parser.add_argument("--forward-type", type=str, required=True, help="Binart type.")
    parser.add_argument("--act-a", default="binary", type=str, help="Binary Func to activation.")
    parser.add_argument("--act-w", default="ste", type=str, help="Binary Func to weight.")

    # AGS
    parser.add_argument("--delta", default=0.01, type=float, help="Lower bound for gradient descent")
    parser.add_argument("--enable_ags", default=False, type=ast.literal_eval)

    # dampen
    parser.add_argument("--enable_dampen", default=False, type=ast.literal_eval)
    parser.add_argument("--dampen_weight", default=0., type=float, help="dampen weight")
    parser.add_argument("--track_momentum", default=0.999, type=float, help="track momentum")
    parser.add_argument("--track_threshold", default=0.0001, type=float, help='track threshold')

    # scaling_factor
    parser.add_argument("--scaling_factor", default=False, type=ast.literal_eval)
    # track_ones
    parser.add_argument("--track_ones", default=False, type=ast.literal_eval)


    ## DDP
    parser.add_argument("--distributed", default=False, type=ast.literal_eval)
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument("-md", '--multiprocessing-distributed', dest="multiprocessing_distributed",
                        type=ast.literal_eval, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    args.logger = None
    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
