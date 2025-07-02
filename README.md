# Custom NeRF Method

A simple custom NeRF (Neural Radiance Fields) implementation for nerfstudio.

## Overview

This is a basic implementation of a NeRF method that includes:

- **Custom Model**: A simple MLP-based neural network for density and color prediction
- **Custom Field**: Neural field representation for 3D scene modeling
- **Custom Pipeline**: Training and inference pipeline
- **Custom Data Manager**: Data loading and preprocessing
- **Custom Data Parser**: Data parsing utilities

## Installation

1. Make sure you have nerfstudio installed:
```bash
pip install nerfstudio
```

2. Install this custom method in editable mode:
```bash
pip install -e .
```

## Usage

After installation, you can use your custom method with nerfstudio:

```bash
# Train on a dataset
ns-train my-method --data /path/to/your/dataset

# View training progress
ns-viewer --load-config /path/to/outputs/.../config.yml

# Render a video
ns-render --load-config /path/to/outputs/.../config.yml --output-path renders/video.mp4
```

## Method Configuration

The method is configured in `my_config.py` with the following default settings:

- **Model**: 8-layer MLP with 256 hidden dimensions
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: Cosine annealing over 30,000 steps
- **Batch size**: 4096 rays per batch
- **Training**: 30,000 iterations

## File Structure

```
custom_nerf/
├── __init__.py              # Package initialization
├── my_config.py             # Method configuration
├── custom_model.py          # Neural network model
├── custom_field.py          # Neural field representation
├── custom_pipeline.py       # Training/inference pipeline
├── custom_datamanager.py    # Data management
├── custom_dataparser.py     # Data parsing
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

## Customization

### Modifying the Model

Edit `custom_model.py` to change the neural network architecture:

```python
# Change the MLP structure
self.mlp = nn.Sequential(
    nn.Linear(3, 512),      # Increase hidden dimension
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 4),      # RGB + density
)
```

### Adding New Losses

In `custom_model.py`, modify the `get_loss_dict` method:

```python
def get_loss_dict(self, outputs, batch, metrics_dict=None):
    # Existing RGB loss
    rgb_loss = self.rgb_loss(pred_rgb, image)
    
    # Add new losses
    depth_loss = self.depth_loss(outputs["depth"], batch["depth"])
    
    loss_dict = {
        "rgb_loss": rgb_loss,
        "depth_loss": depth_loss,
    }
    return loss_dict
```

### Changing Training Parameters

Edit `my_config.py` to modify training settings:

```python
MyMethod = MethodSpecification(
    config=TrainerConfig(
        # ... other settings ...
        max_num_iterations=50000,  # More training iterations
        optimizers={
            "field": {"optimizer": "adam", "lr": 5e-4},  # Different learning rate
            "model": {"optimizer": "adam", "lr": 5e-4},
        },
    ),
)
```

## Features

- **Simple MLP Architecture**: Easy to understand and modify
- **Standard NeRF Rendering**: Volume rendering with ray marching
- **Configurable Parameters**: All hyperparameters can be tuned
- **Integration with nerfstudio**: Full compatibility with nerfstudio ecosystem

## Limitations

This is a basic implementation and may not achieve the same quality as state-of-the-art methods like:
- Instant-NGP
- NeRF++
- Mip-NeRF

For production use, consider:
- Adding positional encoding
- Implementing hierarchical sampling
- Using more sophisticated network architectures
- Adding regularization terms

## Troubleshooting

### Method Not Found
If you get "method not found" errors:
1. Make sure the package is installed: `pip install -e .`
2. Check that the entry point in `pyproject.toml` is correct
3. Restart your terminal/IDE

### Import Errors
If you get import errors:
1. Verify all dependencies are installed
2. Check that file paths in imports are correct
3. Make sure you're in the correct Python environment

## Contributing

To extend this implementation:
1. Modify the relevant files
2. Update the configuration if needed
3. Reinstall the package: `pip install -e .`
4. Test with your dataset

## License

This implementation is provided as-is for educational and research purposes. 