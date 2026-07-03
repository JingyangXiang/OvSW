from utils.sweep import SweepConfig, run_sweep


def _imagenet_extra_args(config: str) -> str:
    if "resnet18" in config:
        return "--batch-size 512 --warmup_length 5"
    return ""


if __name__ == "__main__":
    run_sweep(
        SweepConfig(
            configs=[
                "./configs/largescale/w1a1-ovsw-resnet18-imagenet.yaml",
                "./configs/largescale/w1a1-ovsw-resnet34-imagenet.yaml",
            ],
            epochs=[200],
            dampen_weights=[0.00002],
            deltas=[0.01],
            seeds=[42],
            dataset="ImageNet",
            extra_args_for_config=_imagenet_extra_args,
        )
    )
