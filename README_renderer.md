# NeRF Renderer for Custom Views

This directory contains Python scripts to render new views from your trained NeRF model. You can generate images from arbitrary camera poses by specifying position, look-at point, and other camera parameters.

## Files

- `custom_camera_renderer.py`: Main renderer with custom camera poses and dataset cameras
- `fast_renderer.py`: Fast batch rendering for multiple images
- `exact_nerfstudio_cli.py`: Dataset validation rendering (exact nerfstudio CLI method)
- `simple_renderer.py`: Core utility functions used by other renderers
- `README_renderer.md`: This documentation file

## Quick Start

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

## Function Reference

### `custom_camera_renderer.py`

Main renderer with CLI interface for custom camera poses and dataset cameras.

**Custom Camera Mode:**
```bash
python custom_camera_renderer.py --mode custom \
    --position <x> <y> <z> \
    --look_at <x> <y> <z> \
    --up <x> <y> <z> \
    --fov <degrees> \
    --width <pixels> \
    --height <pixels> \
    --output <filename>
```

**Dataset Camera Mode:**
```bash
python custom_camera_renderer.py --mode dataset \
    --split <train|test|val> \
    --num_images <number> \
    --output_dir <directory>
```

### `fast_renderer.py`

Fast batch renderer that loads the pipeline once and renders multiple images efficiently.

```bash
python fast_renderer.py \
    --num_images <number> \
    --width <pixels> \
    --height <pixels> \
    --output_dir <directory>
```

### `exact_nerfstudio_cli.py`

Exact replication of nerfstudio CLI validation rendering.

```bash
python exact_nerfstudio_cli.py \
    --split <train|test|val> \
    --num_images <number> \
    --output_dir <directory>
```

## Advanced Usage

### Custom Camera Trajectories

You can create custom camera trajectories by varying the position and look_at parameters:

```bash
# Create a circular trajectory
for i in {0..7}; do
    angle=$(echo "2 * 3.14159 * $i / 8" | bc -l)
    x=$(echo "3.0 * c($angle)" | bc -l)
    z=$(echo "3.0 * s($angle)" | bc -l)
    
    python custom_camera_renderer.py --mode custom \
        --position $x 1.0 $z \
        --look_at 0.0 0.0 0.0 \
        --output trajectory_view_$(printf "%03d" $i).png
done
```

### Different Camera Orientations

You can experiment with different up vectors and field of view settings:

```bash
# Top-down view
python custom_camera_renderer.py --mode custom \
    --position 0.0 5.0 0.0 \
    --look_at 0.0 0.0 0.0 \
    --up 0.0 0.0 1.0 \
    --fov 90 \
    --output top_down_view.png

# Wide-angle view
python custom_camera_renderer.py --mode custom \
    --position 2.0 1.0 2.0 \
    --look_at 0.0 0.0 0.0 \
    --fov 120 \
    --output wide_angle_view.png
```

## Performance Optimization

### Rendering Speed

Rendering speed depends heavily on resolution:

- **100x75**: ~0.25s per image ⚡
- **200x150**: ~0.9s per image
- **400x300**: ~8s per image
- **800x600**: ~30s+ per image

### Optimization Tips

1. **Use `fast_renderer.py`** for multiple images (loads pipeline once)
2. **Lower resolution** for quick previews
3. **Higher resolution** for final quality renders
4. **GPU memory**: Monitor usage, reduce resolution if needed

## Tips for Good Results

1. **Camera Distance**: Start with positions around 0.3-1.0 units from scene center
2. **Field of View**: Use 45-90 degrees for most scenes
3. **Look-at Point**: Usually the center of your scene (0, 0, 0) works well
4. **Image Resolution**: Higher resolution for better quality
5. **Up Vector**: Usually (0, 1, 0) works well, but experiment for creative angles

## Troubleshooting

### Common Issues

1. **White lines in images**: Use `exact_nerfstudio_cli.py` for dataset cameras
2. **CUDA out of memory**: Reduce image resolution
3. **Poor quality renders**: Try different camera positions and field of view
4. **Import errors**: Make sure you're in the correct directory and nerfstudio is installed

### Performance Issues

- **Slow rendering**: Use `fast_renderer.py` for multiple images
- **Memory issues**: Reduce resolution or use CPU
- **Pipeline loading**: Takes ~0.5s, but only once per session

## Example Output

The scripts will generate PNG images that you can view with any image viewer. The RGB images show the rendered scene from the specified viewpoint.

### File Organization

```
renders/
├── custom_view.png          # Single custom camera render
├── batch/                   # Fast batch renderer output
│   ├── fast_render_000.png
│   ├── fast_render_001.png
│   └── ...
└── validation/              # Dataset validation output
    ├── test_image_00000.png
    ├── test_image_00001.png
    └── ...
``` 