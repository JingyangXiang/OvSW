import os
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional


@dataclass
class SweepConfig:
    configs: List[str]
    forward_type: str = "xnor"
    epochs: List[int] = field(default_factory=lambda: [200])
    dampen_weights: List[float] = field(default_factory=lambda: [2e-5])
    deltas: List[float] = field(default_factory=lambda: [0.01])
    seeds: List[int] = field(default_factory=lambda: [42])
    enable_ags: bool = True
    enable_dampen: bool = True
    scaling_factor: bool = False
    dataset: Optional[str] = None
    extra_args: str = ""
    extra_args_for_config: Optional[Callable[[str], str]] = None
    shuffle_hyperparams: bool = True


def _build_command(
    config: str,
    forward_type: str,
    name: str,
    epoch: int,
    delta: float,
    seed: int,
    enable_ags: bool,
    enable_dampen: bool,
    dampen_weight: float,
    scaling_factor: bool,
    dataset: Optional[str],
    extra_args: str,
) -> str:
    parts = [
        f"python main.py --config {config}",
        f"--forward-type {forward_type}",
        f"--name {name}",
        f"--epochs {epoch}",
        f"--delta {delta:.5f}",
        f"--seed {seed}",
        f"--enable_ags {enable_ags}",
        f"--enable_dampen {enable_dampen}",
        f"--dampen_weight {dampen_weight}",
        f"--scaling_factor {scaling_factor}",
    ]
    if dataset is not None:
        parts.append(f"--set {dataset}")
    if extra_args:
        parts.append(extra_args.strip())
    return " ".join(parts)


def _run_dir_for_config(config: str, name: str, cuda_available: bool) -> str:
    base = "./runs" if cuda_available else "../../runs"
    config_name = os.path.basename(config).replace(".yaml", "")
    return os.path.join(base, config_name, name)


def run_sweep(
    sweep: SweepConfig,
    runner: Optional[Callable[[str], int]] = None,
    cuda_available: Optional[bool] = None,
) -> None:
    """Run a hyperparameter sweep by invoking main.py for each configuration."""
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()

    if runner is None:
        runner = os.system if cuda_available else print

    dampen_weights = list(sweep.dampen_weights)
    deltas = list(sweep.deltas)
    if sweep.shuffle_hyperparams:
        random.shuffle(dampen_weights)
        random.shuffle(deltas)

    print(dampen_weights)
    print(deltas)

    for config in sweep.configs:
        config_extra_args = sweep.extra_args
        if sweep.extra_args_for_config is not None:
            config_extra_args = f"{config_extra_args} {sweep.extra_args_for_config(config)}".strip()

        for epoch in sweep.epochs:
            for dampen_weight in dampen_weights:
                for delta in deltas:
                    name = f"{sweep.forward_type}_{epoch}_{dampen_weight:.5f}_{delta:.5f}"
                    path = _run_dir_for_config(config, name, cuda_available)

                    if not cuda_available:
                        for seed in sweep.seeds:
                            command = _build_command(
                                config=config,
                                forward_type=sweep.forward_type,
                                name=name,
                                epoch=epoch,
                                delta=delta,
                                seed=seed,
                                enable_ags=sweep.enable_ags,
                                enable_dampen=sweep.enable_dampen,
                                dampen_weight=dampen_weight,
                                scaling_factor=sweep.scaling_factor,
                                dataset=sweep.dataset,
                                extra_args=config_extra_args,
                            )
                            runner(command)
                        continue

                    os.makedirs(path, exist_ok=True)
                    target_runs = len(sweep.seeds) + 3
                    while len(os.listdir(path)) < target_runs:
                        index = max(len(os.listdir(path)) - 3, 0)
                        seed = sweep.seeds[index]
                        command = _build_command(
                            config=config,
                            forward_type=sweep.forward_type,
                            name=name,
                            epoch=epoch,
                            delta=delta,
                            seed=seed,
                            enable_ags=sweep.enable_ags,
                            enable_dampen=sweep.enable_dampen,
                            dampen_weight=dampen_weight,
                            scaling_factor=sweep.scaling_factor,
                            dataset=sweep.dataset,
                            extra_args=config_extra_args,
                        )
                        runner(command)
