# NeRF Renderer for Custom Views

This directory contains Python scripts to render new views from your trained NeRF model. You can generate images from arbitrary camera poses by specifying position, look-at point, and other camera parameters.

## Files

- `simple_renderer.py`: Simple, focused functions for rendering new views
- `render_new_views.py`: More comprehensive renderer class with additional features
- `README_renderer.md`: This documentation file

## Quick Start

### 1. Basic Usage

```python
from simple_renderer import load_trained_model, render_and_save

# Load your trained model
checkpoint_path = "outputs/unnamed/my-method/2025-07-02_200704/nerfstudio_models/step-000029999.ckpt"
model = load_trained_model(checkpoint_path)

# Render a new view
render_and_save(
    model=model,
    position=(2.0, 1.0, 2.0),    # Camera position (x, y, z)
    look_at=(0.0, 0.0, 0.0),     # Point to look at (x, y, z)
    output_path="my_view.png",
    fov=60.0,                    # Field of view in degrees
    image_width=800,
    image_height=600
)
```

### 2. Run Example Script

```bash
python simple_renderer.py
```

This will generate three example images:
- `rendered_view_1.png`: View from position (2, 1, 2) looking at origin
- `rendered_view_2.png`: View from position (-1.5, 0.5, 1.5) looking at origin
- `rendered_view_3.png`: View from above with custom up vector

## Function Reference

### `load_trained_model(checkpoint_path, device="cuda")`

Loads a trained NeRF model from checkpoint.

**Parameters:**
- `checkpoint_path`: Path to the trained model checkpoint
- `device`: Device to run inference on ('cuda' or 'cpu')

**Returns:**
- Loaded CustomModel instance

### `render_view(model, position, look_at, up=(0,1,0), fov=60.0, image_width=800, image_height=600)`

Renders a view from a specific camera pose.

**Parameters:**
- `model`: Loaded CustomModel instance
- `position`: Camera position in world coordinates (x, y, z)
- `look_at`: Point to look at in world coordinates (x, y, z)
- `up`: Up vector (x, y, z), default is (0, 1, 0)
- `fov`: Field of view in degrees, default is 60.0
- `image_width`: Width of the output image, default is 800
- `image_height`: Height of the output image, default is 600

**Returns:**
- Dictionary containing rendered images as numpy arrays:
  - `'rgb'`: RGB image
  - `'depth'`: Depth map
  - `'accumulation'`: Accumulation map

### `render_and_save(model, position, look_at, output_path, ...)`

Renders an image from a pose and saves it to file.

**Parameters:**
- Same as `render_view()` plus:
- `output_path`: Path to save the rendered image

**Returns:**
- Dictionary containing rendered images

## Advanced Usage

### Using the NeRFRenderer Class

For more advanced usage, you can use the `NeRFRenderer` class from `render_new_views.py`:

```python
from render_new_views import NeRFRenderer

# Initialize renderer
renderer = NeRFRenderer("path/to/checkpoint.ckpt")

# Render multiple views
renderer.render_and_save(
    position=(1.0, 0.5, 1.0),
    look_at=(0.0, 0.0, 0.0),
    output_path="custom_view.png",
    fov=45.0,
    image_width=1024,
    image_height=768
)
```

### Custom Camera Trajectories

You can create custom camera trajectories by varying the position and look_at parameters:

```python
import numpy as np

# Create a circular trajectory
radius = 3.0
height = 1.0
num_views = 8

for i in range(num_views):
    angle = 2 * np.pi * i / num_views
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    
    render_and_save(
        model=model,
        position=(x, height, z),
        look_at=(0.0, 0.0, 0.0),
        output_path=f"trajectory_view_{i:03d}.png",
        fov=60.0
    )
```

### Different Camera Orientations

You can experiment with different up vectors and field of view settings:

```python
# Top-down view
render_and_save(
    model=model,
    position=(0.0, 5.0, 0.0),
    look_at=(0.0, 0.0, 0.0),
    up=(0.0, 0.0, 1.0),  # Custom up vector
    output_path="top_down_view.png",
    fov=90.0
)

# Wide-angle view
render_and_save(
    model=model,
    position=(2.0, 1.0, 2.0),
    look_at=(0.0, 0.0, 0.0),
    output_path="wide_angle_view.png",
    fov=120.0  # Very wide field of view
)
```

## Tips for Good Results

1. **Camera Distance**: Start with positions around 1-3 units from the scene center
2. **Field of View**: Use 45-90 degrees for most scenes
3. **Look-at Point**: Usually the center of your scene (0, 0, 0) works well
4. **Image Resolution**: Higher resolution (1024x768 or higher) for better quality
5. **Up Vector**: Usually (0, 1, 0) works well, but experiment for creative angles

## Troubleshooting

### Common Issues

1. **Model not loading**: Make sure the checkpoint path is correct
2. **CUDA out of memory**: Reduce image resolution or use CPU
3. **Poor quality renders**: Try different camera positions and field of view
4. **Import errors**: Make sure you're in the correct directory and nerfstudio is installed

### Performance

- GPU rendering is much faster than CPU
- Higher resolution images take longer to render
- You can batch multiple renders for efficiency

## Example Output

The scripts will generate PNG images that you can view with any image viewer. The RGB images show the rendered scene from the specified viewpoint, while depth and accumulation maps provide additional information about the 3D structure. 