import torch
from torch import Tensor
import numpy as np

import train, setup_training
import jetnet
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinearBounded, FeaturewiseLinear

import datetime

import os
import time
import traceback
import pickle
import json

# model path, type, and number of particles
path = '/graphganvol/vk/g_128D_nh-8_n-2000/outputs/nz_condGD_isab-nz2'
name = path.split('/')[-1]
output_dir = f'/graphganvol/anni/timing'
dataset = '/graphganvol/MPGAN/datasets/'

model = 'gapt'
num_particles = 30
jet_type = 'g'

# real_efps = np.load(f"{mpgan_dir}/efps/{args.jets}.npy")

# models_dir = f"{mpgan_dir}/outputs/{args.name}/models/"
# losses_dir = f"{mpgan_dir}/outputs/{args.name}/losses/"
# jets_dir = f"{mpgan_dir}/outputs/{args.name}/jets/"

# _ = os.system(f"mkdir -p {jets_dir}")

# kpd_path = f"{losses_dir}/kpd.txt"

# prepare jet data
feature_maxes = JetNet.fpnd_norm.feature_maxes
feature_maxes = feature_maxes + [1]

particle_norm = FeaturewiseLinearBounded(
    feature_norms=1.0,
    feature_shifts=[0.0, 0.0, -0.5, -0.5],
    feature_maxes=feature_maxes,
)

jet_norm = FeaturewiseLinear(feature_scales=1.0 / num_particles)
data_args = {
    "jet_type": jet_type,
    "data_dir": dataset,
    "num_particles": num_particles,
    "particle_features": None,
    "jet_features": "num_particles",
    "jet_normalisation": jet_norm,
    "split_fraction": [0.7, 0.3, 0],
}
X = JetNet(**data_args, split="valid")
real_jf = X.jet_data

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gen_multi_batch(
    model_args,
    G: torch.nn.Module,
    batch_size: int,
    num_samples: int,
    num_particles: int,
    out_device: str = "cpu",
    detach: bool = False,
    use_tqdm: bool = True,
    model: str = "mpgan",
    noise: Tensor = None,
    labels: Tensor = None,
    noise_std: float = 0.2,
    save_data: bool = True,
    **extra_args,
) -> Tensor:
    """
    Generates ``num_samples`` jets in batches of ``batch_size``.
    Args are defined in ``gen`` function
    """
    device = next(G.parameters()).device
    G.eval()

    all_labels = Tensor(labels)

    all_noise, _ = train.get_gen_noise(
        model_args, num_samples, num_particles, model, "cpu", noise_std
    )

    start = time.time()
    for i in range((num_samples // batch_size) + 1):
        num_samples_in_batch = min(batch_size, num_samples - (i * batch_size))

        if num_samples_in_batch > 0:
            labels = all_labels[(i * batch_size) : (i * batch_size) + num_samples_in_batch].to(
                device
            )
            noise = all_noise[(i * batch_size) : (i * batch_size) + num_samples_in_batch].to(device)

            gen_data = G(noise, labels)

    end = time.time()

    t = end - start
    print(f"Time for batch size {batch_size}: {t:.2f}s")

    return t


batch_sizes = [
    256,
    512,
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    14336,
    18432,
    22528,
]
timings = []
# load args
with open(f"{path}/{name}_args.txt", "r") as f:
    model_args = setup_training.objectview(eval(f.read()))

G = setup_training.models(model_args, gen_only=True).to("cuda")

G.eval()

num_params = count_trainable_parameters(G)
print(f"Total number of trainable parameters: {num_params}")


for batch_size in batch_sizes:
    print(batch_size)
    try:
        t = gen_jets = gen_multi_batch(
            model_args.__dict__,
            G,
            batch_size,
            50000,
            num_particles,
            model=model,
            labels=real_jf[:50000],
        )
        timings.append([batch_size, t])

    except Exception as e:
        print(f"Error on batch size {batch_size}: {e}")
        traceback.print_exc()
        break


with open(f"{output_dir}/{name}_timing.json", "w") as f:
    json.dump(timings, f)
