#!/usr/bin/env python3
"""
Fast renderer that loads the pipeline once and renders multiple images efficiently.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import time
from simple_renderer import save_image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType

class FastRenderer:
    """Fast renderer that loads pipeline once and renders multiple images."""
    
    def __init__(self, config_path):
        """Initialize the renderer with a loaded pipeline."""
        print(f"Loading pipeline from {config_path}...")
        start_time = time.time()
        _, self.pipeline, _, _ = eval_setup(config_path, test_mode="inference")
        load_time = time.time() - start_time
        print(f"Pipeline loaded in {load_time:.2f}s")
        print(f"Model device: {self.pipeline.device}")
        
        # Set model to eval mode for faster inference
        self.pipeline.model.eval()
        
    def create_camera(self, position, look_at, up=(0, 1, 0), fov=60.0, 
                     image_width=800, image_height=600):
        """Create a camera object."""
        device = self.pipeline.device
        
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
    
    def render_single(self, position, look_at, up=(0, 1, 0), fov=60.0,
                     image_width=800, image_height=600, output_path=None):
        """Render a single image."""
        start_time = time.time()
        
        # Create camera
        camera = self.create_camera(position, look_at, up, fov, image_width, image_height)
        
        # Render
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(camera)
        
        # Get RGB output
        rgb = outputs['rgb'].cpu().numpy()
        
        render_time = time.time() - start_time
        
        # Save if path provided
        if output_path:
            save_image(rgb, output_path)
            print(f"  Saved: {output_path}")
        
        print(f"  Render time: {render_time:.3f}s")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]  mean: {rgb.mean():.3f}")
        print(f"  Image shape: {rgb.shape}")
        
        return rgb
    
    def render_multiple(self, camera_configs, output_dir="renders/fast"):
        """Render multiple images efficiently."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_start = time.time()
        
        for i, config in enumerate(camera_configs):
            print(f"Rendering image {i+1}/{len(camera_configs)}...")
            output_path = output_dir / f"fast_render_{i:03d}.png"
            
            self.render_single(
                position=config['position'],
                look_at=config['look_at'],
                up=config.get('up', (0, 1, 0)),
                fov=config.get('fov', 60.0),
                image_width=config.get('width', 400),
                image_height=config.get('height', 300),
                output_path=str(output_path)
            )
        
        total_time = time.time() - total_start
        print(f"\nTotal render time: {total_time:.2f}s")
        print(f"Average per image: {total_time/len(camera_configs):.2f}s")
        print(f"Images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fast renderer for multiple images.")
    parser.add_argument('--config', type=str, 
                       default="outputs/unnamed/my-method/2025-07-02_200704/config.yml",
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='renders/fast',
                       help='Output directory')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to render')
    parser.add_argument('--width', type=int, default=400,
                       help='Image width')
    parser.add_argument('--height', type=int, default=300,
                       help='Image height')
    
    args = parser.parse_args()
    
    # Initialize renderer
    renderer = FastRenderer(Path(args.config))
    
    # Create multiple camera configurations
    camera_configs = []
    
    # Different positions around the scene
    positions = [
        (0.3, 0.3, 0.3),
        (-0.2, 0.4, 0.2),
        (0.0, 0.0, 0.5),
        (0.4, -0.1, 0.3),
        (0.1, 0.5, 0.1),
    ]
    
    for i in range(min(args.num_images, len(positions))):
        camera_configs.append({
            'position': positions[i],
            'look_at': (0.0, 0.0, 0.0),
            'up': (0.0, 1.0, 0.0),
            'fov': 60.0,
            'width': args.width,
            'height': args.height,
        })
    
    # Render all images
    renderer.render_multiple(camera_configs, args.output_dir)

if __name__ == "__main__":
    main() 