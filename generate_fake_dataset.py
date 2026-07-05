#!/usr/bin/env python3
"""Generate paired latent vectors and images from the real model snapshot."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
STYLEGAN_DIR = REPO_ROOT / "stylegan2-ada-pytorch"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "real_model_snapshot"


def find_snapshot(snapshot_dir: Path) -> Path:
    snapshots = sorted(snapshot_dir.glob("network-snapshot-*.pkl"))
    if not snapshots:
        raise FileNotFoundError(f'No "network-snapshot-*.pkl" found in {snapshot_dir}')
    return snapshots[-1]


def prepare_output_dir(root: Path, n: int) -> Path:
    outdir = root / f"fake_dataset_{n}"
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "latents").mkdir(parents=True, exist_ok=True)
    return outdir


def save_pair(outdir: Path, index: int, latent: np.ndarray, image: np.ndarray) -> None:
    stem = f"{index:06d}"
    np.save(outdir / "latents" / f"{stem}.npy", latent)
    Image.fromarray(image, "RGB").save(outdir / "images" / f"{stem}.png")


def write_metadata(
    outdir: Path,
    *,
    snapshot: Path,
    n: int,
    seed: int,
    batch_size: int,
    truncation_psi: float,
    noise_mode: str,
    class_idx: Optional[int],
) -> None:
    metadata = {
        "snapshot": str(snapshot),
        "n": n,
        "seed": seed,
        "batch_size": batch_size,
        "truncation_psi": truncation_psi,
        "noise_mode": noise_mode,
        "class_idx": class_idx,
        "image_dir": "images",
        "latent_dir": "latents",
        "image_format": "png",
        "latent_format": "npy",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_generator(snapshot: Path, device: str):
    if str(STYLEGAN_DIR) not in sys.path:
        sys.path.insert(0, str(STYLEGAN_DIR))

    import dnnlib
    import legacy

    print(f'Loading network from "{snapshot}"...')
    with dnnlib.util.open_url(str(snapshot)) as f:
        return legacy.load_network_pkl(f)["G_ema"].to(device).eval()


def make_labels(G, batch_size: int, class_idx: Optional[int], device: str):
    import torch

    labels = torch.zeros([batch_size, G.c_dim], device=device)
    if G.c_dim == 0:
        if class_idx is not None:
            print("warn: --class-idx ignored for an unconditional network")
        return labels

    if class_idx is None:
        raise ValueError("This snapshot is conditional; pass --class-idx.")
    if class_idx < 0 or class_idx >= G.c_dim:
        raise ValueError(f"--class-idx must be between 0 and {G.c_dim - 1}")
    labels[:, class_idx] = 1
    return labels


def generate_dataset(
    *,
    n: int,
    snapshot_dir: Path,
    output_root: Path,
    seed: int,
    batch_size: int,
    truncation_psi: float,
    noise_mode: str,
    device: str,
    class_idx: Optional[int],
) -> Path:
    if n <= 0:
        raise ValueError("--n must be greater than zero")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    import torch

    snapshot = find_snapshot(snapshot_dir)
    outdir = prepare_output_dir(output_root, n)
    G = load_generator(snapshot, device)
    rng = np.random.RandomState(seed)

    write_metadata(
        outdir,
        snapshot=snapshot,
        n=n,
        seed=seed,
        batch_size=batch_size,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        class_idx=class_idx,
    )

    generated = 0
    while generated < n:
        current_batch = min(batch_size, n - generated)
        z_np = rng.randn(current_batch, G.z_dim).astype(np.float32)
        z = torch.from_numpy(z_np).to(device)
        labels = make_labels(G, current_batch, class_idx, device)

        with torch.no_grad():
            images = G(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
            images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images_np = images.cpu().numpy()

        for offset in range(current_batch):
            sample_idx = generated + offset
            save_pair(outdir, sample_idx, z_np[offset], images_np[offset])

        generated += current_batch
        print(f"Generated {generated}/{n} samples", flush=True)

    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fake dataset of latent-vector/image pairs from real_model_snapshot."
    )
    parser.add_argument("--n", type=int, default=50000, help="Number of images to generate.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trunc", dest="truncation_psi", type=float, default=1.0)
    parser.add_argument("--noise-mode", choices=["const", "random", "none"], default="const")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--class-idx", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = generate_dataset(
        n=args.n,
        snapshot_dir=args.snapshot_dir,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        truncation_psi=args.truncation_psi,
        noise_mode=args.noise_mode,
        device=args.device,
        class_idx=args.class_idx,
    )
    print(f'Done. Wrote fake dataset to "{outdir}".')


if __name__ == "__main__":
    main()
