#!/usr/bin/env python3
"""
Custom camera renderer using the exact nerfstudio CLI method.
Supports both dataset cameras and custom camera poses.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from simple_renderer import save_image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType

def create_custom_camera(position, look_at, up=(0, 1, 0), fov=60.0, 
                        image_width=800, image_height=600, device="cuda"):
    """
    Create a custom camera from position, look_at point, and other parameters.
    
    Args:
        position: Camera position (x, y, z)
        look_at: Point to look at (x, y, z)
        up: Up vector (x, y, z)
        fov: Field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels
        device: Device to create tensors on
        
    Returns:
        Cameras object
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    position = torch.tensor(position, dtype=torch.float32, device=device)
    look_at = torch.tensor(look_at, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    
    # Calculate camera orientation
    forward = look_at - position
    forward = forward / torch.norm(forward)
    
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    
    up = torch.cross(right, forward)
    up = up / torch.norm(up)
    
    # Create rotation matrix
    rotation = torch.stack([right, up, -forward], dim=0)
    
    # Create camera to world transform
    camera_to_world = torch.eye(4, dtype=torch.float32, device=device)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = position
    
    # Convert to (1, 3, 4) shape for Cameras
    camera_to_world = camera_to_world[:3, :].unsqueeze(0)
    
    # Calculate focal length from FOV
    focal_length = image_width / (2 * torch.tan(torch.tensor(fov * np.pi / 360)))
    
    # Create camera parameters
    fx = fy = focal_length
    cx, cy = image_width / 2, image_height / 2
    
    # Create Cameras object
    camera = Cameras(
        camera_to_worlds=camera_to_world,
        fx=torch.tensor([fx], device=device),
        fy=torch.tensor([fy], device=device),
        cx=torch.tensor([cx], device=device),
        cy=torch.tensor([cy], device=device),
        width=torch.tensor([image_width]),
        height=torch.tensor([image_height]),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    
    return camera

def render_custom_camera(pipeline, position, look_at, up=(0, 1, 0), fov=60.0,
                        image_width=800, image_height=600, output_path="custom_render.png"):
    """
    Render from a custom camera pose using the nerfstudio pipeline.
    
    Args:
        pipeline: Loaded nerfstudio pipeline
        position: Camera position (x, y, z)
        look_at: Point to look at (x, y, z)
        up: Up vector (x, y, z)
        fov: Field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels
        output_path: Path to save the rendered image
    """
    print(f"Rendering custom camera...")
    print(f"  Position: {position}")
    print(f"  Look at: {look_at}")
    print(f"  FOV: {fov}°")
    print(f"  Resolution: {image_width}x{image_height}")
    
    # Create custom camera
    camera = create_custom_camera(
        position=position,
        look_at=look_at,
        up=up,
        fov=fov,
        image_width=image_width,
        image_height=image_height,
        device=pipeline.device
    )
    
    # Render using the exact same method as nerfstudio CLI
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)
    
    # Get RGB output
    rgb = outputs['rgb'].cpu().numpy()
    
    # Save image
    save_image(rgb, output_path)
    
    print(f"  Saved: {output_path}")
    print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]  mean: {rgb.mean():.3f}")
    print(f"  Image shape: {rgb.shape}")
    
    return rgb

def render_dataset_cameras(pipeline, split="test", num_images=4, output_dir="renders/dataset"):
    """Render from dataset cameras (same as exact_nerfstudio_cli.py)."""
    
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
    parser = argparse.ArgumentParser(description="Render using custom camera poses or dataset cameras.")
    parser.add_argument('--mode', type=str, default='custom', choices=['custom', 'dataset'], 
                       help='Rendering mode: custom camera pose or dataset cameras')
    
    # Custom camera arguments
    parser.add_argument('--position', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                       help='Camera position (x y z)')
    parser.add_argument('--look_at', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help='Point to look at (x y z)')
    parser.add_argument('--up', type=float, nargs=3, default=[0.0, 1.0, 0.0],
                       help='Up vector (x y z)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Field of view in degrees')
    parser.add_argument('--width', type=int, default=800,
                       help='Image width in pixels')
    parser.add_argument('--height', type=int, default=600,
                       help='Image height in pixels')
    parser.add_argument('--output', type=str, default='custom_render.png',
                       help='Output image path')
    
    # Dataset camera arguments
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val'],
                       help='Dataset split to render (for dataset mode)')
    parser.add_argument('--num_images', type=int, default=4,
                       help='Number of images to render (for dataset mode)')
    parser.add_argument('--output_dir', type=str, default='renders/dataset',
                       help='Output directory (for dataset mode)')
    
    args = parser.parse_args()
    
    # Load pipeline
    config_path = Path("outputs/unnamed/my-method/2025-07-02_200704/config.yml")
    print(f"Loading pipeline from {config_path}...")
    _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
    print(f"Pipeline loaded successfully")
    print(f"Model device: {pipeline.device}")
    
    if args.mode == 'custom':
        # Render custom camera
        render_custom_camera(
            pipeline=pipeline,
            position=tuple(args.position),
            look_at=tuple(args.look_at),
            up=tuple(args.up),
            fov=args.fov,
            image_width=args.width,
            image_height=args.height,
            output_path=args.output
        )
    else:
        # Render dataset cameras
        render_dataset_cameras(
            pipeline=pipeline,
            split=args.split,
            num_images=args.num_images,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 