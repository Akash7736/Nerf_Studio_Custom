#!/usr/bin/env python3
"""
Exact replication of nerfstudio CLI validation rendering.
This uses the same pipeline setup and model.get_outputs_for_camera method as the CLI.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from simple_renderer import save_image
from nerfstudio.utils.eval_utils import eval_setup

def render_exact_nerfstudio_cli(split="test", num_images=4, output_dir="renders/exact_cli"):
    """Render using the exact same method as nerfstudio CLI."""
    
    # Load the exact same pipeline as nerfstudio CLI
    config_path = Path("outputs/unnamed/my-method/2025-07-02_200704/config.yml")
    
    print(f"Loading pipeline from {config_path}...")
    _, pipeline, _, _ = eval_setup(
        config_path,
        test_mode="inference",
    )
    
    print(f"Pipeline loaded successfully")
    print(f"Model device: {pipeline.device}")
    
    # Get the dataset based on split
    if split == "train":
        dataset = pipeline.datamanager.train_dataset
    else:  # test/val
        dataset = pipeline.datamanager.eval_dataset
    
    print(f"Using {split} dataset with {len(dataset)} images")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render images using the exact same method as CLI
    for i in range(min(num_images, len(dataset))):
        print(f"Rendering {split} image {i}...")
        camera = dataset.cameras[i:i+1]  # Cameras object, shape (1, ...)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        rgb = outputs['rgb'].cpu().numpy()
        out_path = output_dir / f"{split}_image_{i:05d}.png"
        save_image(rgb, str(out_path))
        print(f"  Saved: {out_path}")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]  mean: {rgb.mean():.3f}")
        print(f"  Image shape: {rgb.shape}")
    print(f"\nRendered {min(num_images, len(dataset))} images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Render using exact nerfstudio CLI method.")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val'], help='Dataset split to render')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to render')
    parser.add_argument('--output_dir', type=str, default='renders/exact_cli', help='Output directory')
    args = parser.parse_args()
    
    render_exact_nerfstudio_cli(split=args.split, num_images=args.num_images, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 