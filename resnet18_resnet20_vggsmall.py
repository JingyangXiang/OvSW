from utils.sweep import SweepConfig, run_sweep

if __name__ == "__main__":
    run_sweep(
        SweepConfig(
            configs=[
                "./configs/smallscale/w1a1-ovsw-resnet20-cifar10.yaml",
            ],
            epochs=[300, 400, 500, 600],
            dampen_weights=[0.0009],
            deltas=[0.04],
            seeds=[46, 47, 48, 49, 50],
        )
    )
