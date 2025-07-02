#!/usr/bin/env python3
"""
Simple NeRF renderer for generating new views from a trained model.
Provides easy-to-use functions for rendering images from arbitrary camera poses.
"""

import torch
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
from PIL import Image

# Import nerfstudio components
from nerfstudio.cameras.cameras import Cameras, CameraType

# Import your custom model
import sys
sys.path.append('.')
from my_method.custom_model import CustomModel, CustomModelConfig


def load_trained_model(checkpoint_path: str, device: str = "cuda") -> CustomModel:
    """
    Load a trained NeRF model from checkpoint.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Loaded CustomModel instance
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict - the state dict is directly in pipeline
    model_state = {}
    for key, value in checkpoint["pipeline"].items():
        if key.startswith("_model."):
            # Remove the "_model." prefix
            new_key = key[7:]  # Remove "_model." prefix
            model_state[new_key] = value
    
    # Create model config
    config = CustomModelConfig(
        num_layers=8,
        hidden_dim=256,
        out_dim=4,
    )
    
    # Create model instance with required parameters
    from nerfstudio.data.scene_box import SceneBox
    
    # Create a default scene box (you may need to adjust this based on your scene)
    scene_box = SceneBox(aabb=torch.tensor([[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]], device=device))
    num_train_data = 100  # Default value, adjust if needed
    
    model = CustomModel(config=config, scene_box=scene_box, num_train_data=num_train_data)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {checkpoint_path}")
    print(f"Using device: {device}")
    
    return model


def create_camera(position: Tuple[float, float, float],
                 look_at: Tuple[float, float, float],
                 up: Tuple[float, float, float] = (0, 1, 0),
                 fov: float = 60.0,
                 image_width: int = 800,
                 image_height: int = 600,
                 device: str = "cuda") -> Cameras:
    """
    Create a camera from position, look_at point, and up vector.
    
    Args:
        position: Camera position in world coordinates (x, y, z)
        look_at: Point to look at in world coordinates (x, y, z)
        up: Up vector (x, y, z)
        fov: Field of view in degrees
        image_width: Width of the output image
        image_height: Height of the output image
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
    
    # Create rotation matrix (camera coordinate system)
    # In camera space: +X is right, +Y is up, -Z is forward
    rotation = torch.stack([right, up, -forward], dim=0)
    
    # Create camera to world transform
    camera_to_world = torch.eye(4, dtype=torch.float32, device=device)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = position
    
    # nerfstudio expects (N, 3, 4) shape
    camera_to_world_3x4 = camera_to_world[:3, :].unsqueeze(0)
    
    # Create camera parameters - use focal length based on FOV
    focal_length = image_width / (2 * torch.tan(torch.tensor(fov * np.pi / 360)))
    fx = fy = focal_length
    cx, cy = image_width / 2, image_height / 2
    
    # Create cameras object
    cameras = Cameras(
        camera_to_worlds=camera_to_world_3x4,
        fx=torch.tensor([fx], dtype=torch.float32, device=device),
        fy=torch.tensor([fy], dtype=torch.float32, device=device),
        cx=torch.tensor([cx], dtype=torch.float32, device=device),
        cy=torch.tensor([cy], dtype=torch.float32, device=device),
        width=torch.tensor([image_width]),
        height=torch.tensor([image_height]),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    
    return cameras


def render_view(model: CustomModel,
                position: Tuple[float, float, float],
                look_at: Tuple[float, float, float],
                up: Tuple[float, float, float] = (0, 1, 0),
                fov: float = 60.0,
                image_width: int = 800,
                image_height: int = 600) -> Dict[str, np.ndarray]:
    """
    Render a view from a specific camera pose.
    
    Args:
        model: Loaded CustomModel instance
        position: Camera position in world coordinates (x, y, z)
        look_at: Point to look at in world coordinates (x, y, z)
        up: Up vector (x, y, z)
        fov: Field of view in degrees
        image_width: Width of the output image
        image_height: Height of the output image
        
    Returns:
        Dictionary containing rendered images as numpy arrays
    """
    # Create camera
    cameras = create_camera(
        position=position,
        look_at=look_at,
        up=up,
        fov=fov,
        image_width=image_width,
        image_height=image_height,
        device=next(model.parameters()).device
    )
    
    # Render image
    with torch.no_grad():
        ray_bundle = cameras.generate_rays(camera_indices=0)
        outputs = model(ray_bundle)
    
    # Convert to numpy arrays
    result = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu().numpy()
    
    return result


def save_image(image: np.ndarray, filename: str):
    """Save an image to file."""
    # Ensure image is in correct format (0-1 range, RGB)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert to PIL and save
    pil_image = Image.fromarray(image)
    pil_image.save(filename)
    print(f"Image saved to {filename}")


def render_and_save(model: CustomModel,
                   position: Tuple[float, float, float],
                   look_at: Tuple[float, float, float],
                   output_path: str,
                   up: Tuple[float, float, float] = (0, 1, 0),
                   fov: float = 60.0,
                   image_width: int = 800,
                   image_height: int = 600) -> Dict[str, np.ndarray]:
    """
    Render an image from a pose and save it to file.
    
    Args:
        model: Loaded CustomModel instance
        position: Camera position in world coordinates (x, y, z)
        look_at: Point to look at in world coordinates (x, y, z)
        output_path: Path to save the rendered image
        up: Up vector (x, y, z)
        fov: Field of view in degrees
        image_width: Width of the output image
        image_height: Height of the output image
        
    Returns:
        Dictionary containing rendered images
    """
    outputs = render_view(
        model=model,
        position=position,
        look_at=look_at,
        up=up,
        fov=fov,
        image_width=image_width,
        image_height=image_height
    )
    
    # Save RGB image
    rgb_image = outputs['rgb']
    save_image(rgb_image, output_path)
    
    return outputs


# Example usage functions
def example_usage():
    """Example of how to use the renderer functions."""
    
    # Path to your trained model checkpoint
    checkpoint_path = "outputs/unnamed/my-method/2025-07-02_200704/nerfstudio_models/step-000029999.ckpt"
    
    # Load the trained model
    model = load_trained_model(checkpoint_path)
    
    # Example 1: Render from a specific position looking at origin
    print("Rendering example view 1...")
    render_and_save(
        model=model,
        position=(0.3, 0.3, 0.3),  # Camera position (tighter to scene)
        look_at=(0.0, 0.0, 0.0),   # Look at origin
        output_path="rendered_view_1.png",
        fov=60.0,
        image_width=200,
        image_height=150
    )
    
    # Example 2: Render from a different angle
    print("Rendering example view 2...")
    render_and_save(
        model=model,
        position=(-1.5, 0.5, 1.5),
        look_at=(0.0, 0.0, 0.0),
        output_path="rendered_view_2.png",
        fov=45.0,
        image_width=200,
        image_height=150
    )
    
    # Example 3: Render with custom up vector
    print("Rendering example view 3...")
    render_and_save(
        model=model,
        position=(0.0, 2.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),  # Custom up vector
        output_path="rendered_view_3.png",
        fov=90.0,
        image_width=200,
        image_height=150
    )
    
    print("All example renders completed!")


if __name__ == "__main__":
    example_usage() 