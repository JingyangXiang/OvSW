import pathlib
import time

CSV_HEADER = (
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

CSV_ROW_TEMPLATE = (
    "{now}, "
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
)


def write_result_to_csv(arch, **kwargs):
    """Append a training run summary to runs/train_{arch}_from_scratch.csv."""
    results = pathlib.Path("runs") / f"train_{arch}_from_scratch.csv"
    if not results.exists():
        results.write_text(CSV_HEADER)

    now = time.strftime("%m-%d-%y_%H:%M:%S")
    with open(results, "a+", encoding="utf-8") as f:
        f.write(CSV_ROW_TEMPLATE.format(now=now, arch=arch, **kwargs))


# Backward-compatible alias for the original misspelled function name.
write_result_to_csv_scrach = write_result_to_csv
