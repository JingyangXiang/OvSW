import os
import random

import torch

run = print
if torch.cuda.is_available():
    run = os.system
configs = ["./configs/largescale/w1a1-ovsw-resnet18-imagenet.yaml",
           "./configs/largescale/w1a1-ovsw-resnet34-imagenet.yaml"]
forward_type = "xnor"
set = "ImageNet"
save_every = 1
epochs = [200, ]
scaling_factor = False
enable_ags = True
enable_dampen = True
dampen_weights = [0.00002, ]
random.shuffle(dampen_weights)
print(dampen_weights)

seeds = [42, ]
nums = len(seeds)
deltas = [0.01, ]
random.shuffle(deltas)
print(deltas)

for config in configs:
    for epoch in epochs:
        for dampen_weight in dampen_weights:
            for delta in deltas:
                if not torch.cuda.is_available():
                    name = f'{forward_type}_{epoch}_{dampen_weight:.5f}_{delta:.5f}'
                    path = os.path.join("../../runs", os.path.basename(config).replace('.yaml', ''), name)
                    for seed in seeds:
                        orders = (f"python main.py --config {config} --forward-type {forward_type} --name {name} \\"
                                  f"--epochs {epoch} --delta {delta:.5f} --seed {seed} --enable_ags {enable_ags} \\"
                                  f"--enable_dampen {enable_dampen} --dampen_weight {dampen_weight} --set {set} \\"
                                  f"--scaling_factor {scaling_factor}")
                        if 'resnet18' in orders:
                            orders += " --batch-size 512 --warmup_length 5"
                        run(orders)
                else:
                    name = f'{forward_type}_{epoch}_{dampen_weight:.5f}_{delta:.5f}'
                    path = os.path.join("./runs", os.path.basename(config).replace('.yaml', ''), name)
                    os.makedirs(path, exist_ok=True)
                    while len(os.listdir(path)) < nums + 3:
                        index = max(len(os.listdir(path)) - 3, 0)
                        seed = seeds[index]
                        orders = (f"python main.py --config {config} --forward-type {forward_type} --name {name} \\"
                                  f"--epochs {epoch} --delta {delta:.5f} --seed {seed} --enable_ags {enable_ags} \\"
                                  f"--enable_dampen {enable_dampen} --dampen_weight {dampen_weight} --set {set}  \\"
                                  f"--scaling_factor {scaling_factor}")
                        if 'resnet18' in orders:
                            orders += " --batch-size 512 --warmup_length 5"
                        run(orders)