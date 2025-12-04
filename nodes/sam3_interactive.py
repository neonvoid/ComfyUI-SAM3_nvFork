"""
SAM3 Point Collector - Interactive Point Selection

Point editor widget adapted from ComfyUI-KJNodes
Original: https://github.com/kijai/ComfyUI-KJNodes
Author: kijai
License: Apache 2.0

Modifications for SAM3:
- Removed bounding box functionality
- Simplified to positive/negative points only
- Outputs point arrays for use with SAM3Segmentation node
"""

import torch
import numpy as np
import json
import io
import base64
from PIL import Image


class SAM3PointCollector:
    """
    Interactive Point Collector for SAM3

    Displays image canvas in the node where users can click to add:
    - Positive points (Left-click) - green circles
    - Negative points (Shift+Left-click or Right-click) - red circles

    Outputs point arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Left-click to add positive points (green), Shift+Left-click or Right-click to add negative points (red). Points are automatically normalized to image dimensions."
                }),
                "points_store": ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points")
    FUNCTION = "collect_points"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, points_store, coordinates, neg_coordinates):
        # Return hash based on actual point content, not object identity
        # This ensures downstream nodes don't re-run when points haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        result = h.hexdigest()
        print(f"[IS_CHANGED DEBUG] SAM3PointCollector: shape={image.shape}, coords={coordinates}, neg_coords={neg_coordinates}")
        print(f"[IS_CHANGED DEBUG] SAM3PointCollector: returning hash={result}")
        return result

    def collect_points(self, image, points_store, coordinates, neg_coordinates):
        """
        Collect points from interactive canvas

        Supports both single-object (legacy) and multi-object formats.

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            points_store: Combined JSON storage (hidden widget) - may contain 'objects' key for multi-object
            coordinates: Positive points JSON (hidden widget) - legacy format
            neg_coordinates: Negative points JSON (hidden widget) - legacy format

        Returns:
            Tuple of (positive_points, negative_points) as separate SAM3_POINTS_PROMPT outputs
            For multi-object mode, positive_points will contain an 'objects' key with per-object data
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(points_store.encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3PointCollector._cache:
            cached = SAM3PointCollector._cache[cache_key]
            print(f"[SAM3 Point Collector] CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        print(f"[SAM3 Point Collector] CACHE MISS - computing new result for key={cache_key[:8]}")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        print(f"[SAM3 Point Collector] Image dimensions: {img_width}x{img_height}")

        # Parse points_store to check for multi-object format
        try:
            store_data = json.loads(points_store) if points_store and points_store.strip() else {}
        except json.JSONDecodeError:
            store_data = {}

        # Check for multi-object format (new v9 widget)
        if "objects" in store_data and len(store_data["objects"]) > 0:
            # MULTI-OBJECT MODE
            print(f"[SAM3 Point Collector] Multi-object mode: {len(store_data['objects'])} objects detected")

            # Build multi-object output format
            multi_objects = []
            total_pos = 0
            total_neg = 0

            for obj_data in store_data["objects"]:
                obj_id = obj_data.get("obj_id", 1)
                pos_pts = obj_data.get("positive_points", [])
                neg_pts = obj_data.get("negative_points", [])

                # Normalize coordinates to 0-1
                normalized_pos = []
                for pt in pos_pts:
                    # Points from JS are [x, y] arrays
                    if isinstance(pt, list) and len(pt) >= 2:
                        normalized_pos.append([pt[0] / img_width, pt[1] / img_height])

                normalized_neg = []
                for pt in neg_pts:
                    if isinstance(pt, list) and len(pt) >= 2:
                        normalized_neg.append([pt[0] / img_width, pt[1] / img_height])

                multi_objects.append({
                    "obj_id": obj_id,
                    "positive_points": normalized_pos,
                    "negative_points": normalized_neg
                })

                total_pos += len(normalized_pos)
                total_neg += len(normalized_neg)
                print(f"[SAM3 Point Collector]   Object {obj_id}: {len(normalized_pos)} positive, {len(normalized_neg)} negative points")

            print(f"[SAM3 Point Collector] Multi-object output: {len(multi_objects)} objects, {total_pos} total positive, {total_neg} total negative")

            # Output format: positive_points contains 'objects' key for multi-object mode
            # negative_points is empty dict (all negatives are in objects)
            positive_points = {"objects": multi_objects}
            negative_points = {"points": [], "labels": []}  # Empty - negatives are per-object

        else:
            # LEGACY SINGLE-OBJECT MODE
            # Parse coordinates from JSON
            try:
                pos_coords = json.loads(coordinates) if coordinates and coordinates.strip() else []
                neg_coords = json.loads(neg_coordinates) if neg_coordinates and neg_coordinates.strip() else []
            except json.JSONDecodeError:
                pos_coords = []
                neg_coords = []

            print(f"[SAM3 Point Collector] Legacy mode: {len(pos_coords)} positive, {len(neg_coords)} negative points")

            # Convert to SAM3 point format - separate positive and negative outputs
            # SAM3 expects normalized coordinates (0-1), so divide by image dimensions
            positive_points = {"points": [], "labels": []}
            negative_points = {"points": [], "labels": []}

            # Add positive points (label = 1) - normalize to 0-1
            for p in pos_coords:
                normalized_x = p['x'] / img_width
                normalized_y = p['y'] / img_height
                positive_points["points"].append([normalized_x, normalized_y])
                positive_points["labels"].append(1)
                print(f"[SAM3 Point Collector]   Positive point: ({p['x']:.1f}, {p['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

            # Add negative points (label = 0) - normalize to 0-1
            for n in neg_coords:
                normalized_x = n['x'] / img_width
                normalized_y = n['y'] / img_height
                negative_points["points"].append([normalized_x, normalized_y])
                negative_points["labels"].append(0)
                print(f"[SAM3 Point Collector]   Negative point: ({n['x']:.1f}, {n['y']:.1f}) -> ({normalized_x:.3f}, {normalized_y:.3f})")

            print(f"[SAM3 Point Collector] Legacy output: {len(positive_points['points'])} positive, {len(negative_points['points'])} negative")

        # Cache the result
        result = (positive_points, negative_points)
        SAM3PointCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


class SAM3BBoxCollector:
    """
    Interactive BBox Collector for SAM3

    Displays image canvas in the node where users can click and drag to add:
    - Positive bounding boxes (Left-click and drag) - cyan rectangles
    - Negative bounding boxes (Shift+Left-click and drag or Right-click and drag) - red rectangles

    Outputs bbox arrays to feed into SAM3Segmentation node.
    """
    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to display in interactive canvas. Click and drag to draw positive bboxes (cyan), Shift+Click/Right-click and drag to draw negative bboxes (red). Bounding boxes are automatically normalized to image dimensions."
                }),
                "bboxes": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_bboxes": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("positive_bboxes", "negative_bboxes")
    FUNCTION = "collect_bboxes"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, bboxes, neg_bboxes):
        # Return hash based on actual bbox content, not object identity
        # This ensures downstream nodes don't re-run when bboxes haven't changed
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        result = h.hexdigest()
        print(f"[IS_CHANGED DEBUG] SAM3BBoxCollector: shape={image.shape}, bboxes={bboxes}, neg_bboxes={neg_bboxes}")
        print(f"[IS_CHANGED DEBUG] SAM3BBoxCollector: returning hash={result}")
        return result

    def collect_bboxes(self, image, bboxes, neg_bboxes):
        """
        Collect bounding boxes from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            bboxes: Positive BBoxes JSON array (hidden widget)
            neg_bboxes: Negative BBoxes JSON array (hidden widget)

        Returns:
            Tuple of (positive_bboxes, negative_bboxes) as separate SAM3_BOXES_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3BBoxCollector._cache:
            cached = SAM3BBoxCollector._cache[cache_key]
            print(f"[SAM3 BBox Collector] CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_base64]},
                "result": cached  # Return the SAME objects
            }

        print(f"[SAM3 BBox Collector] CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse bboxes from JSON
        try:
            pos_bbox_list = json.loads(bboxes) if bboxes and bboxes.strip() else []
            neg_bbox_list = json.loads(neg_bboxes) if neg_bboxes and neg_bboxes.strip() else []
        except json.JSONDecodeError:
            pos_bbox_list = []
            neg_bbox_list = []

        print(f"[SAM3 BBox Collector] Collected {len(pos_bbox_list)} positive, {len(neg_bbox_list)} negative bboxes")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        print(f"[SAM3 BBox Collector] Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3_BOXES_PROMPT format with boxes and labels
        positive_boxes = []
        positive_labels = []
        negative_boxes = []
        negative_labels = []

        # Add positive bboxes (label = True)
        for bbox in pos_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            positive_boxes.append([center_x, center_y, width, height])
            positive_labels.append(True)  # Positive boxes
            print(f"[SAM3 BBox Collector]   Positive BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        # Add negative bboxes (label = False)
        for bbox in neg_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox['x1'] / img_width
            y1_norm = bbox['y1'] / img_height
            x2_norm = bbox['x2'] / img_width
            y2_norm = bbox['y2'] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            negative_boxes.append([center_x, center_y, width, height])
            negative_labels.append(False)  # Negative boxes
            print(f"[SAM3 BBox Collector]   Negative BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})")

        print(f"[SAM3 BBox Collector] Output: {len(positive_boxes)} positive, {len(negative_boxes)} negative bboxes")

        # Format as SAM3_BOXES_PROMPT (dict with 'boxes' and 'labels' keys)
        positive_prompt = {
            "boxes": positive_boxes,
            "labels": positive_labels
        }
        negative_prompt = {
            "boxes": negative_boxes,
            "labels": negative_labels
        }

        # Cache the result
        result = (positive_prompt, negative_prompt)
        SAM3BBoxCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {
            "ui": {"bg_image": [img_base64]},
            "result": result
        }

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SAM3PointCollector": SAM3PointCollector,
    "SAM3BBoxCollector": SAM3BBoxCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PointCollector": "SAM3 Point Collector",
    "SAM3BBoxCollector": "SAM3 BBox Collector 📦",
}
