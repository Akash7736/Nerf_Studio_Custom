"""Custom NeRF model implementation."""

from typing import Dict, Optional, Tuple, Type
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colors, misc


@dataclass
class CustomModelConfig(ModelConfig):
    """Custom NeRF model config."""
    _target: Type = field(default_factory=lambda: CustomModel)
    num_layers: int = 8
    hidden_dim: int = 256
    out_dim: int = 4


class CustomField(Field):
    """Custom NeRF field."""
    
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Create MLP layers
        layers = []
        in_dim = 3  # xyz coordinates
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, ray_samples, **kwargs):
        """Forward pass."""
        positions = ray_samples.frustums.get_positions()
        
        # Forward pass through MLP
        outputs = self.mlp(positions)
        
        # Split outputs into RGB and density
        rgb = torch.sigmoid(outputs[..., :3])
        density = F.relu(outputs[..., 3:4])
        
        return {
            FieldHeadNames.RGB: rgb,
            FieldHeadNames.DENSITY: density,
        }


class CustomModel(Model):
    """Custom NeRF model."""

    config: CustomModelConfig

    def __init__(
        self,
        config: CustomModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        
        # Create field
        self.field = CustomField(
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            out_dim=self.config.out_dim,
        )
        
        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        
        # Losses
        self.rgb_loss = MSELoss()

    def get_param_groups(self) -> Dict[str, list]:
        """Get parameter groups for the optimizer."""
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def forward(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Flatten rays if needed
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        if origins.dim() > 2:
            origins = origins.reshape(-1, 3)
            directions = directions.reshape(-1, 3)
        # Sample points along rays using uniform sampling
        num_samples = 64  # Number of samples per ray
        near = 0.1
        far = 10.0
        
        # Create uniform samples along the ray
        t_vals = torch.linspace(0., 1., steps=num_samples, device=origins.device)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand(origins.shape[0], num_samples)
        
        # Get sample points along rays
        origins_exp = origins.unsqueeze(1).expand(-1, num_samples, -1)
        directions_exp = directions.unsqueeze(1).expand(-1, num_samples, -1)
        sample_points = origins_exp + z_vals.unsqueeze(-1) * directions_exp
        
        # Create a simple forward pass through the field
        # Reshape to batch of points
        points_flat = sample_points.reshape(-1, 3)
        
        # Forward pass through MLP
        outputs = self.field.mlp(points_flat)
        
        # Split outputs into RGB and density
        rgb = torch.sigmoid(outputs[..., :3])
        density = F.relu(outputs[..., 3:4])
        
        # Reshape back to ray structure
        rgb = rgb.reshape(origins.shape[0], num_samples, 3)
        density = density.reshape(origins.shape[0], num_samples, 1)
        
        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
        
        # Convert density to alpha
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)
        
        # Compute weights for volume rendering
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]
        
        # Render outputs
        rgb_rendered = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        depth = torch.sum(weights * z_vals, dim=1)
        accumulation = torch.sum(weights, dim=1)
        
        outputs = {
            "rgb": rgb_rendered,
            "depth": depth,
            "accumulation": accumulation,
        }
        
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Get loss dictionary."""
        image = batch["image"].to(self.device)
        pred_rgb = outputs["rgb"]
        
        rgb_loss = self.rgb_loss(pred_rgb, image)
        
        loss_dict = {"rgb_loss": rgb_loss}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get image metrics and images for evaluation."""
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = outputs["accumulation"]
        
        # Calculate metrics
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(rgb, image)
        
        # Create images dict
        images_dict = {
            "img": image,
            "rgb": rgb,
            "acc": acc,
        }
        
        return metrics_dict, images_dict
    
    def psnr(self, rgb: torch.Tensor, image: torch.Tensor) -> float:
        """Calculate PSNR between predicted and ground truth images."""
        mse = torch.mean((rgb - image) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
