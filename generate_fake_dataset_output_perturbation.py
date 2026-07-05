#!/usr/bin/env python3
"""Generate latent/image pairs with mild output perturbation."""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import generate_fake_dataset as base


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "real_model_snapshot"


def prepare_output_dir(root: Path, n: int) -> Path:
    outdir = root / f"fake_dataset_output_perturbation_{n}"
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "latents").mkdir(parents=True, exist_ok=True)
    return outdir


def apply_output_perturbation(
    image: np.ndarray,
    rng: np.random.RandomState,
    *,
    perturb_std: float,
) -> np.ndarray:
    if perturb_std < 0:
        raise ValueError("--perturb-std must be non-negative")
    if perturb_std == 0:
        return image.copy()

    noise = rng.normal(loc=0.0, scale=perturb_std, size=image.shape)
    perturbed = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(perturbed, 0, 255).astype(np.uint8)


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
    perturb_std: float,
    perturb_seed: int,
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
        "perturbation": {
            "type": "gaussian_pixel_noise",
            "std": perturb_std,
            "seed": perturb_seed,
            "description": "Mild Gaussian noise added to generated RGB pixels before saving.",
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


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
    perturb_std: float,
    perturb_seed: int,
) -> Path:
    if n <= 0:
        raise ValueError("--n must be greater than zero")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if perturb_std < 0:
        raise ValueError("--perturb-std must be non-negative")

    import torch

    snapshot = base.find_snapshot(snapshot_dir)
    outdir = prepare_output_dir(output_root, n)
    G = base.load_generator(snapshot, device)
    latent_rng = np.random.RandomState(seed)
    perturb_rng = np.random.RandomState(perturb_seed)

    write_metadata(
        outdir,
        snapshot=snapshot,
        n=n,
        seed=seed,
        batch_size=batch_size,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        class_idx=class_idx,
        perturb_std=perturb_std,
        perturb_seed=perturb_seed,
    )

    generated = 0
    while generated < n:
        current_batch = min(batch_size, n - generated)
        z_np = latent_rng.randn(current_batch, G.z_dim).astype(np.float32)
        z = torch.from_numpy(z_np).to(device)
        labels = base.make_labels(G, current_batch, class_idx, device)

        with torch.no_grad():
            images = G(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
            images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images_np = images.cpu().numpy()

        for offset in range(current_batch):
            sample_idx = generated + offset
            perturbed = apply_output_perturbation(
                images_np[offset],
                perturb_rng,
                perturb_std=perturb_std,
            )
            save_pair(outdir, sample_idx, z_np[offset], perturbed)

        generated += current_batch
        print(f"Generated {generated}/{n} perturbed samples", flush=True)

    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate latent-vector/image pairs with mild output perturbation."
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
    parser.add_argument("--perturb-std", type=float, default=3.0)
    parser.add_argument("--perturb-seed", type=int, default=12345)
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
        perturb_std=args.perturb_std,
        perturb_seed=args.perturb_seed,
    )
    print(f'Done. Wrote perturbed fake dataset to "{outdir}".')


if __name__ == "__main__":
    main()
