# StyleGAN2-ADA Impressionist Cluster 1

This repository contains a StyleGAN2-ADA PyTorch checkout, a prepared dataset at
`datasets/impressionist_cluster1_256.zip`, training outputs, and scripts for
generating clean and output-perturbed fake datasets from a trained snapshot.

Start from the repository root:

```bash
cd /path/to/ex3
```

## Environment Setup

You need a working NVIDIA driver, Conda, and a CUDA-capable GPU. The StyleGAN2-ADA
code compiles custom CUDA extensions, so the environment must include both PyTorch
with CUDA support and `nvcc`.

### Modern CUDA Setup

Use this setup for current GPUs, including RTX 40/50-series cards:

```bash
conda create -n sg2ada-rtx50 python=3.10 -y
conda activate sg2ada-rtx50

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install click requests tqdm pyspng ninja imageio-ffmpeg psutil scipy pillow imageio "numpy<2"

conda install -y -c nvidia cuda-toolkit=12.8
```

Check the environment:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

nvcc --version
nvidia-smi
```

### Older CUDA 11 Setup

For older GPUs where the original StyleGAN2-ADA stack is preferred:

```bash
conda create -n sg2ada python=3.7 -y
conda activate sg2ada

python -m pip install "pip<24.1" setuptools wheel
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
python -m pip install numpy==1.21.6 scipy==1.7.3 pillow==9.5.0 psutil==5.9.8 imageio==2.31.2

conda install -c conda-forge cudatoolkit-dev=11.7 -y
```

## Training

The dataset used for this run is already expected at:

```text
datasets/impressionist_cluster1_256.zip
```

Validate the training command without starting a full run:

```bash
cd stylegan2-ada-pytorch

python train.py \
  --outdir=../training-runs \
  --data=../datasets/impressionist_cluster1_256.zip \
  --gpus=1 \
  --cfg=paper256 \
  --mirror=1 \
  --aug=noaug \
  --metrics=none \
  --snap=10 \
  --dry-run
```

Start the 1000 kimg training run:

```bash
cd stylegan2-ada-pytorch

python train.py \
  --outdir=../training-runs \
  --data=../datasets/impressionist_cluster1_256.zip \
  --gpus=1 \
  --cfg=paper256 \
  --mirror=1 \
  --aug=noaug \
  --metrics=none \
  --snap=10 \
  --kimg=1000
```

Training snapshots are written under `training-runs/<run-name>/` as
`network-snapshot-*.pkl`.

## Preparing a Snapshot for Dataset Generation

The fake dataset scripts look for the model snapshot in:

```text
real_model_snapshot/
```

Copy the snapshot you want to use into that directory. For example:

```bash
cd /path/to/ex3
mkdir -p real_model_snapshot
cp training-runs/<run-name>/network-snapshot-001000.pkl real_model_snapshot/
```

If there are multiple `network-snapshot-*.pkl` files in `real_model_snapshot/`,
the scripts use the latest one by filename.

## Generate a Clean Fake Dataset

From the repository root:

```bash
cd /path/to/ex3
conda activate sg2ada-rtx50

python generate_fake_dataset.py \
  --n 50000 \
  --batch-size 32 \
  --device cuda
```

This writes:

```text
fake_dataset_50000/
  images/000000.png
  latents/000000.npy
  metadata.json
```

Each `.npy` file stores the latent vector used for the matching PNG image.

## Generate an Output-Perturbed Fake Dataset

The perturbed generator uses the same model and latent/image layout, but adds
mild Gaussian pixel noise before saving each image. The default perturbation is
small enough to preserve visual quality.

```bash
cd /path/to/ex3
conda activate sg2ada-rtx50

python generate_fake_dataset_output_perturbation.py \
  --n 50000 \
  --batch-size 32 \
  --device cuda \
  --perturb-std 3.0 \
  --perturb-seed 12345
```

This writes:

```text
fake_dataset_output_perturbation_50000/
  images/000000.png
  latents/000000.npy
  metadata.json
```

`--perturb-std` is measured on the 0-255 RGB pixel scale. Keep it low, around
`1.0` to `5.0`, if the images should remain visually acceptable.

## Convert Images to a StyleGAN2-ADA Dataset

StyleGAN2-ADA expects training data as an image folder or dataset ZIP. Use
`dataset_tool.py` to convert a folder of images into a compatible ZIP.

To recreate the real Impressionist cluster dataset from a raw image folder:

```bash
cd /path/to/ex3/stylegan2-ada-pytorch

python dataset_tool.py \
  --source=../datasets/impressionist_cluster1_raw \
  --dest=../datasets/impressionist_cluster1_256.zip \
  --transform=center-crop \
  --width=256 \
  --height=256
```

To convert the clean generated images into a StyleGAN2-ADA dataset:

```bash
cd /path/to/ex3/stylegan2-ada-pytorch

python dataset_tool.py \
  --source=../fake_dataset_50000/images \
  --dest=../datasets/fake_dataset_50000_256.zip \
  --transform=center-crop \
  --width=256 \
  --height=256
```

To convert the output-perturbed generated images:

```bash
cd /path/to/ex3/stylegan2-ada-pytorch

python dataset_tool.py \
  --source=../fake_dataset_output_perturbation_50000/images \
  --dest=../datasets/fake_dataset_output_perturbation_50000_256.zip \
  --transform=center-crop \
  --width=256 \
  --height=256
```

## Train on the Fake Datasets

After converting the generated images to StyleGAN2-ADA ZIP datasets, train on
them with the same settings as the real dataset. The only differences are the
`--data` path and `--metrics=fid50k_full`.

Train on the clean generated dataset:

```bash
cd /path/to/ex3/stylegan2-ada-pytorch

python train.py \
  --outdir=../training-runs \
  --data=../datasets/fake_dataset_50000_256.zip \
  --gpus=1 \
  --cfg=paper256 \
  --mirror=1 \
  --aug=noaug \
  --metrics=fid50k_full \
  --snap=10 \
  --kimg=1000
```

Train on the output-perturbed generated dataset:

```bash
cd /path/to/ex3/stylegan2-ada-pytorch

python train.py \
  --outdir=../training-runs \
  --data=../datasets/fake_dataset_output_perturbation_50000_256.zip \
  --gpus=1 \
  --cfg=paper256 \
  --mirror=1 \
  --aug=noaug \
  --metrics=fid50k_full \
  --snap=10 \
  --kimg=1000
```

The FID values are written into each training run directory as
`metric-fid50k_full.jsonl`.

## Evaluate Trained Fake Models

Use `evaluate_model.py` from the repository root to compute FID for a trained
fake model. The script always evaluates `--snapshot`; the reference is either a
dataset passed with `--dataset` or a model passed with `--model`. These two
reference flags are mutually exclusive. Each run writes a JSON result file to
`evaluation/`.

Accuracy measures FID between a fake model and the real dataset:

```bash
cd /path/to/ex3

python evaluate_model.py \
  --snapshot training-runs/<fake-model-run>/network-snapshot-001000.pkl \
  --dataset datasets/impressionist_cluster1_256.zip \
  --num-images 50000 \
  --batch-size 64 \
  --device cuda \
  --workers 0 \
  --output-dir evaluation
```

Fidelity measures FID between a fake model and newly generated images from the
real model:

```bash
cd /path/to/ex3

python evaluate_model.py \
  --snapshot training-runs/<fake-model-run>/network-snapshot-001000.pkl \
  --model real_model_snapshot/network-snapshot-001000.pkl \
  --num-images 50000 \
  --batch-size 64 \
  --device cuda \
  --workers 0 \
  --output-dir evaluation
```

The `--model` mode generates reference images in memory and does not write them
to disk. Keep `--workers 0` when the reference is a ZIP dataset; multiprocessing
workers can trigger ZIP read errors. Add `--json` if you also want the result
printed to stdout as JSON.
