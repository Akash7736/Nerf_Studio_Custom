# Custom NeRF Implementation

A custom NeRF (Neural Radiance Fields) implementation integrated with the nerfstudio framework, featuring a complete training pipeline and custom rendering tools.

## Overview

This project includes:

- **Custom NeRF Model**: MLP-based neural network for density and color prediction
- **Training Pipeline**: Complete training and inference pipeline integrated with nerfstudio
- **Custom Renderers**: Python scripts for rendering new views from trained models
- **Dataset Processing**: Tools for creating custom datasets from videos

## Installation

1. Make sure you have nerfstudio installed:
```bash
pip install nerfstudio
```

2. Install this custom method in editable mode:
```bash
pip install -e .
```

## Training

### Using Pre-existing Datasets

Train on existing datasets (like the IUI3-RedSea dataset used in this project):

```bash
# Train on a dataset
ns-train my-method --data /path/to/your/dataset

# View training progress
ns-viewer --load-config /path/to/outputs/.../config.yml

# Render validation images
ns-render --load-config /path/to/outputs/.../config.yml --output-path renders/validation.mp4
```

### Creating Custom Datasets from Videos

Process your own videos into nerfstudio-compatible datasets:

```bash
# Basic video processing
ns-process-data video \
    --data your_video.mp4 \
    --output-dir custom_datasets/my_video_dataset \
    --num-frames-target 300 \
    --matching-method sequential \
    --sfm-tool colmap \
    --feature-type sift \
    --matcher-type NN

# Train on your custom dataset
ns-train my-method --data custom_datasets/my_video_dataset
```

**Video Requirements for Best Results:**
- Slow, smooth camera movement
- Good lighting and rich textures
- 5-30 seconds duration
- Avoid fast motion, motion blur, or reflective surfaces

## Custom Rendering

After training, use the custom renderer scripts to generate new views:

### 1. Custom Camera Poses

Render from arbitrary camera positions:

```bash
# Basic custom camera
python custom_camera_renderer.py --mode custom \
    --position 0.3 0.3 0.3 \
    --look_at 0.0 0.0 0.0 \
    --fov 60 \
    --width 400 \
    --height 300 \
    --output my_view.png

# Different angle with narrow FOV
python custom_camera_renderer.py --mode custom \
    --position -0.2 0.4 0.2 \
    --look_at 0.0 0.0 0.0 \
    --fov 45 \
    --width 800 \
    --height 600
```

### 2. Fast Batch Rendering

Render multiple images efficiently:

```bash
# Render 10 images quickly
python fast_renderer.py \
    --num_images 10 \
    --width 200 \
    --height 150 \
    --output_dir renders/batch
```

### 3. Dataset Validation

Render using the same cameras as training/validation:

```bash
# Render test split cameras
python exact_nerfstudio_cli.py \
    --split test \
    --num_images 4 \
    --output_dir renders/validation
```

## Method Configuration

The custom method uses the following default settings:

- **Model**: 8-layer MLP with 256 hidden dimensions
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: Cosine annealing over 30,000 steps
- **Batch size**: 4096 rays per batch
- **Training**: 30,000 iterations
- **Scene Box**: [-0.25, -0.25, -0.25] to [0.25, 0.25, 0.25]
- **Near/Far Planes**: 2.0 to 6.0

## File Structure

```
custom_nerf/
├── my_method/                    # Custom method package
│   ├── __init__.py              # Package initialization
│   ├── custom_model.py          # Neural network model
│   └── my_config.py             # Method configuration
├── custom_camera_renderer.py    # Main renderer with custom poses
├── fast_renderer.py             # Fast batch rendering
├── exact_nerfstudio_cli.py      # Dataset validation rendering
├── simple_renderer.py           # Core utility functions
├── README_renderer.md           # Renderer documentation
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Performance Optimization

### Rendering Speed

Rendering speed depends heavily on resolution:

- **100x75**: ~0.25s per image ⚡
- **200x150**: ~0.9s per image
- **400x300**: ~8s per image
- **800x600**: ~30s+ per image

### Training Performance

- **GPU Memory**: ~8GB recommended for 400x300 resolution
- **Training Time**: ~2-4 hours for 30,000 iterations
- **Convergence**: PSNR typically stabilizes around 20,000 iterations

## Features

- **Complete NeRF Implementation**: Full volume rendering with ray marching
- **nerfstudio Integration**: Seamless integration with nerfstudio ecosystem
- **Custom Rendering Tools**: Flexible camera pose control
- **Dataset Processing**: Video-to-dataset conversion
- **Performance Optimized**: Fast batch rendering capabilities
- **Configurable**: All hyperparameters can be tuned

## Example Results

This implementation has been successfully tested on:
- **IUI3-RedSea Dataset**: Underwater scene reconstruction
- **Custom Video Datasets**: User-generated video content

Training typically achieves:
- **PSNR**: 20-25 dB
- **SSIM**: 0.7-0.8
- **LPIPS**: 0.1-0.2

## Troubleshooting

### Training Issues
- **CUDA OOM**: Reduce batch size or image resolution
- **Poor Convergence**: Check learning rate and scene box settings
- **Method Not Found**: Ensure package is installed with `pip install -e .`

### Rendering Issues
- **White Lines**: Use the exact nerfstudio CLI method (`exact_nerfstudio_cli.py`)
- **Slow Rendering**: Reduce resolution or use `fast_renderer.py`
- **Poor Quality**: Increase resolution or adjust camera parameters

### Dataset Issues
- **COLMAP Failures**: Use sequential matching for videos
- **Poor Reconstruction**: Ensure video has good lighting and texture
- **Import Errors**: Check file paths and dependencies

## Contributing

To extend this implementation:
1. Modify the relevant files in `my_method/`
2. Update configuration if needed
3. Reinstall: `pip install -e .`
4. Test with your dataset

## License

This implementation is provided as-is for educational and research purposes. 