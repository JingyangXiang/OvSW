import os
import pathlib
import shutil

import torch


def init_model(model, logger, args):
    # 0. init logger
    for module in model.modules():
        if hasattr(module, "init_logger"):
            module.init_logger(logger=logger)

    # 1. init forward type
    for module in model.modules():
        if hasattr(module, "init_forward_type"):
            module.init_forward_type(forward_type=args.forward_type)

    # 2. init act_a and act_w
    for module in model.modules():
        if hasattr(module, "init_binary_activation_and_weight"):
            module.init_binary_activation_and_weight(act_a=args.act_a, act_w=args.act_w)

    # 3. init enable_oscillation_dampen
    for module in model.modules():
        if hasattr(module, "init_dampen"):
            module.init_dampen(args=args)

    # 4. init adaptive_gradient_scale
    for module in model.modules():
        if hasattr(module, "init_adaptive_gradient_scale"):
            module.init_adaptive_gradient_scale(args=args)

    # 5. init scaling factor
    for module in model.modules():
        if hasattr(module, "init_scaling_factor"):
            module.init_scaling_factor(scaling_factor=args.scaling_factor)

    # 6. init epoch
    for module in model.modules():
        if hasattr(module, "init_epoch"):
            module.init_epoch(start_epoch=args.start_epoch, end_epoch=args.epochs, epoch_num=args.epochs)


def disable_oscillation_dampen(model):
    for module in model.modules():
        if hasattr(module, "disable_oscillation_dampen"):
            module.disable_oscillation_dampen()


def update_epoch(model, epoch):
    for module in model.modules():
        if hasattr(module, "update_epoch"):
            module.update_epoch(current_epoch=epoch)


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


def get_parameters_num(model, logger):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = param.numel()  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    logger.info(f'Total params: {Total_params}')
    logger.info(f'Trainable params: {Trainable_params}')
    logger.info(f'Non-trainable params: {NonTrainable_params}')
