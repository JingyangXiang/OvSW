def apply_ovsw_grad_updates(model, args, epoch: int, log_epoch_metrics: bool = False) -> None:
    """Apply OvSW adaptive gradient scaling and silent-weight dampening."""
    for module in model.modules():
        if hasattr(module, "adaptive_gradient_scale"):
            ratio = module.adaptive_gradient_scale()
            if log_epoch_metrics and args.enable_ags:
                args.logger.info(f"Epoch [{epoch}]: {ratio.detach().cpu().item():.5f}")
        if hasattr(module, "conditional_dampening"):
            ratio = module.conditional_dampening()
            if log_epoch_metrics and args.enable_dampen and ratio is not None:
                args.logger.info(
                    f"Epoch [{epoch}]: {ratio.detach().cpu().item() * 100:.5f}%"
                )
