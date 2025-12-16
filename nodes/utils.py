"""
Utility functions for ComfyUI-SAM3 nodes
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


# =============================================================================
# Color Mapping for Visualization
# =============================================================================

# Named color mapping (RGB, 0-1 range)
COLOR_MAP = {
    # Primary colors
    "red": [1.0, 0.3, 0.3],
    "green": [0.3, 1.0, 0.3],
    "blue": [0.0, 0.5, 1.0],
    # Secondary colors
    "yellow": [1.0, 1.0, 0.3],
    "magenta": [1.0, 0.3, 1.0],
    "cyan": [0.3, 1.0, 1.0],
    # Tertiary colors
    "orange": [1.0, 0.6, 0.3],
    "purple": [0.6, 0.3, 1.0],
    "pink": [1.0, 0.5, 0.7],
    "lime": [0.5, 1.0, 0.3],
    "teal": [0.3, 0.8, 0.8],
    "coral": [1.0, 0.5, 0.5],
    # Additional colors
    "gold": [1.0, 0.84, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "navy": [0.0, 0.0, 0.5],
    "maroon": [0.5, 0.0, 0.0],
    "olive": [0.5, 0.5, 0.0],
    "aqua": [0.0, 1.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5],
    "grey": [0.5, 0.5, 0.5],  # British spelling alias
}

# Default fallback colors (used when user colors run out)
DEFAULT_COLORS = [
    [0.0, 0.5, 1.0],   # Blue
    [1.0, 0.3, 0.3],   # Red
    [0.3, 1.0, 0.3],   # Green
    [1.0, 1.0, 0.3],   # Yellow
    [1.0, 0.3, 1.0],   # Magenta
    [0.3, 1.0, 1.0],   # Cyan
    [1.0, 0.6, 0.3],   # Orange
    [0.6, 0.3, 1.0],   # Purple
]


def parse_color_string(color_string):
    """
    Parse a comma-separated color string into a list of RGB colors.

    Supports:
    - Named colors: "red, blue, green"
    - Hex colors: "#FF0000, #00FF00"
    - RGB tuples: "1.0,0.5,0.0" (single color) or "red,blue" (multiple)

    Args:
        color_string: Comma-separated color names or values

    Returns:
        List of RGB colors (0-1 range), or empty list if invalid
    """
    if not color_string or not color_string.strip():
        return []

    colors = []
    # Split by comma, but be careful with RGB tuples
    parts = [p.strip().lower() for p in color_string.split(',')]

    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if it's a named color
        if part in COLOR_MAP:
            colors.append(COLOR_MAP[part])
            i += 1
            continue

        # Check if it's a hex color
        if part.startswith('#'):
            try:
                hex_color = part.lstrip('#')
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    colors.append([r, g, b])
            except ValueError:
                pass  # Invalid hex, skip
            i += 1
            continue

        # Check if it's an RGB tuple (three consecutive float values)
        try:
            if i + 2 < len(parts):
                r = float(parts[i])
                g = float(parts[i + 1])
                b = float(parts[i + 2])
                if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
                    colors.append([r, g, b])
                    i += 3
                    continue
        except ValueError:
            pass

        # Unknown color, skip
        i += 1

    return colors


def get_color_palette(color_string, num_objects):
    """
    Get a color palette for visualization.

    Args:
        color_string: User-specified colors (comma-separated names/hex)
        num_objects: Number of objects to color

    Returns:
        List of RGB colors (0-1 range) with length >= num_objects
    """
    user_colors = parse_color_string(color_string)

    if not user_colors:
        # No user colors, use defaults
        return DEFAULT_COLORS

    # If user specified enough colors, use them
    if len(user_colors) >= num_objects:
        return user_colors

    # Extend with defaults for remaining objects
    result = list(user_colors)
    for i in range(len(user_colors), num_objects):
        # Cycle through defaults for extra objects
        result.append(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

    return result


# =============================================================================
# File/Path Utilities
# =============================================================================

def get_comfy_models_dir():
    """Get the ComfyUI models directory"""
    # Try to find ComfyUI root by going up from custom_nodes
    current = Path(__file__).parent.parent.absolute()  # ComfyUI-SAM3
    comfy_custom_nodes = current.parent  # custom_nodes
    comfy_root = comfy_custom_nodes.parent  # ComfyUI root

    models_dir = comfy_root / "models" / "sam3"
    models_dir.mkdir(parents=True, exist_ok=True)

    return str(models_dir)


def comfy_image_to_pil(image):
    """
    Convert ComfyUI image tensor to PIL Image

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        PIL Image
    """
    # ComfyUI images are [B, H, W, C] in range [0, 1]
    if isinstance(image, torch.Tensor):
        # Take first image if batch
        if image.dim() == 4:
            image = image[0]

        # Convert to numpy
        img_np = image.cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        img_np = (img_np * 255).astype(np.uint8)

        # Convert to PIL
        pil_image = Image.fromarray(img_np)
        return pil_image

    return image


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor

    Args:
        pil_image: PIL Image

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert to numpy array
    img_np = np.array(pil_image).astype(np.float32)

    # Normalize to [0, 1]
    img_np = img_np / 255.0

    # Convert to tensor [H, W, C]
    img_tensor = torch.from_numpy(img_np)

    # Add batch dimension [1, H, W, C]
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def masks_to_comfy_mask(masks):
    """
    Convert SAM3 masks to ComfyUI mask format

    Args:
        masks: torch.Tensor [N, H, W] or [N, 1, H, W] binary masks

    Returns:
        ComfyUI mask tensor [N, H, W] in range [0, 1] on CPU
    """
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Move to CPU to ensure compatibility with downstream nodes
        return masks.cpu()
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Already on CPU since from numpy
        return masks

    return masks


def visualize_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """
    Create visualization of masks overlaid on image

    Args:
        image: PIL Image or numpy array
        masks: torch.Tensor [N, H, W] binary masks
        boxes: Optional torch.Tensor [N, 4] bounding boxes in [x0, y0, x1, y1]
        scores: Optional torch.Tensor [N] confidence scores
        alpha: Transparency of mask overlay

    Returns:
        PIL Image with visualization
    """
    if isinstance(image, torch.Tensor):
        image = comfy_image_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))

    # Convert to numpy for processing
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize masks to image size if needed
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Create colored overlay
    np.random.seed(42)  # Consistent colors
    overlay = img_np.copy()

    for i, mask in enumerate(masks_np):
        # Squeeze extra dimensions (masks may be [1, H, W] or [H, W])
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Resize mask to image size if needed
        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Random color for this mask
        color = np.random.rand(3)

        # Apply colored mask
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0.5,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    # Convert back to PIL
    result = Image.fromarray((overlay * 255).astype(np.uint8))

    # Draw boxes if provided
    if boxes is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box

            # Random color for this box (same seed for consistency)
            np.random.seed(42 + i)
            color_int = tuple((np.random.rand(3) * 255).astype(int).tolist())

            # Draw box
            draw.rectangle([x0, y0, x1, y1], outline=color_int, width=3)

            # Draw score if provided
            if scores is not None:
                score = scores[i] if isinstance(scores, (list, np.ndarray)) else scores[i].item()
                text = f"{score:.2f}"
                draw.text((x0, y0 - 15), text, fill=color_int)

    return result


def tensor_to_list(tensor):
    """Convert torch tensor to python list"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor


from contextlib import contextmanager
import gc


@contextmanager
def inference_context():
    """
    Context manager ensuring cleanup after inference.

    Usage:
        with inference_context():
            # ... inference code ...

    This ensures gc.collect() and torch.cuda.empty_cache() are called
    after inference, even if an exception occurs.
    """
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_gpu_memory():
    """
    Force GPU memory cleanup.

    Call this after inference to ensure VRAM is freed.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
