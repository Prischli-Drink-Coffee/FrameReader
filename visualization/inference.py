"""
Specialized inference result visualization.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, inference visualization will be disabled")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceVisualizer:
    """Specialized visualizer for inference results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default visualization settings
        self.settings = {
            'bbox_colors': {
                'high_confidence': 'green',
                'medium_confidence': 'orange', 
                'low_confidence': 'red',
                'default': 'blue'
            },
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.5
            },
            'font_sizes': {
                'large': 20,
                'medium': 16,
                'small': 12
            }
        }
    
    def visualize_ocr_result(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prediction: str,
        ground_truth: Optional[str] = None,
        bounding_boxes: Optional[List[Dict]] = None,
        confidence: Optional[float] = None,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Visualize OCR inference result with text and bounding boxes."""
        
        # Convert input image to PIL
        if isinstance(image, np.ndarray):
            if OPENCV_AVAILABLE and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        # Load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", self.settings['font_sizes']['large'])
            font_medium = ImageFont.truetype("arial.ttf", self.settings['font_sizes']['medium'])
            font_small = ImageFont.truetype("arial.ttf", self.settings['font_sizes']['small'])
        except (OSError, IOError):
            font_large = font_medium = font_small = ImageFont.load_default()
        
        # Draw bounding boxes
        if bounding_boxes:
            for bbox in bounding_boxes:
                self._draw_bounding_box(draw, bbox, font_small)
        
        # Create text overlay
        viz_image = self._add_text_overlay(
            viz_image, text_prediction, ground_truth, confidence,
            font_large, font_medium
        )
        
        # Save if path provided
        if save_path:
            viz_image.save(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return viz_image
    
    def _draw_bounding_box(
        self, 
        draw: ImageDraw.ImageDraw, 
        bbox: Dict[str, Any], 
        font: ImageFont.ImageFont
    ) -> None:
        """Draw a single bounding box with confidence score."""
        
        # Extract coordinates
        if 'coords' in bbox:
            x1, y1, x2, y2 = bbox['coords']
        elif 'bbox' in bbox:
            x1, y1, x2, y2 = bbox['bbox']
        else:
            # Try alternative formats
            x1 = bbox.get('x1', bbox.get('left', 0))
            y1 = bbox.get('y1', bbox.get('top', 0))
            x2 = bbox.get('x2', bbox.get('right', x1 + bbox.get('width', 0)))
            y2 = bbox.get('y2', bbox.get('bottom', y1 + bbox.get('height', 0)))
        
        # Determine color based on confidence
        bbox_confidence = bbox.get('confidence', 1.0)
        if bbox_confidence >= self.settings['confidence_thresholds']['high']:
            color = self.settings['bbox_colors']['high_confidence']
        elif bbox_confidence >= self.settings['confidence_thresholds']['medium']:
            color = self.settings['bbox_colors']['medium_confidence']
        else:
            color = self.settings['bbox_colors']['low_confidence']
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw confidence score
        if bbox_confidence < 1.0:  # Only show if not default confidence
            conf_text = f"{bbox_confidence:.2f}"
            text_bbox = draw.textbbox((0, 0), conf_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Background for confidence text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                fill=color, outline=color
            )
            draw.text((x1 + 2, y1 - text_height - 2), conf_text, fill='white', font=font)
        
        # Draw detected text if available
        detected_text = bbox.get('text', '')
        if detected_text:
            # Truncate long text
            if len(detected_text) > 20:
                detected_text = detected_text[:17] + "..."
            
            draw.text((x1, y2 + 2), detected_text, fill=color, font=font)
    
    def _add_text_overlay(
        self,
        image: Image.Image,
        prediction: str,
        ground_truth: Optional[str],
        confidence: Optional[float],
        font_large: ImageFont.ImageFont,
        font_medium: ImageFont.ImageFont
    ) -> Image.Image:
        """Add text overlay at the bottom of the image."""
        
        img_width, img_height = image.size
        
        # Calculate overlay height based on content
        lines = []
        
        # Prediction text (wrap if too long)
        pred_text = f"Prediction: {prediction}"
        if len(pred_text) > 80:
            pred_text = pred_text[:77] + "..."
        lines.append(('prediction', pred_text, 'blue'))
        
        # Ground truth if available
        if ground_truth is not None:
            gt_text = f"Ground Truth: {ground_truth}"
            if len(gt_text) > 80:
                gt_text = gt_text[:77] + "..."
            lines.append(('ground_truth', gt_text, 'green'))
        
        # Confidence if available
        if confidence is not None:
            conf_text = f"Confidence: {confidence:.3f}"
            lines.append(('confidence', conf_text, 'red'))
        
        # Calculate overlay dimensions
        line_height = 30
        padding = 10
        overlay_height = len(lines) * line_height + 2 * padding
        
        # Create overlay
        overlay = Image.new('RGB', (img_width, overlay_height), color='white')
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw text lines
        for i, (line_type, text, color) in enumerate(lines):
            y_pos = padding + i * line_height
            font = font_large if line_type == 'prediction' else font_medium
            overlay_draw.text((padding, y_pos), text, fill=color, font=font)
        
        # Combine image and overlay
        combined = Image.new('RGB', (img_width, img_height + overlay_height))
        combined.paste(image, (0, 0))
        combined.paste(overlay, (0, img_height))
        
        return combined
    
    def compare_predictions(
        self,
        image: Image.Image,
        predictions: Dict[str, str],
        ground_truth: Optional[str] = None,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Compare predictions from multiple models."""
        
        img_width, img_height = image.size
        
        # Calculate layout
        num_models = len(predictions)
        line_height = 25
        padding = 10
        header_height = 30
        
        overlay_height = header_height + (num_models + (1 if ground_truth else 0)) * line_height + 2 * padding
        
        # Create comparison overlay
        overlay = Image.new('RGB', (img_width, overlay_height), color='white')
        draw = ImageDraw.Draw(overlay)
        
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            header_font = ImageFont.truetype("arial.ttf", 18)
        except (OSError, IOError):
            font = header_font = ImageFont.load_default()
        
        # Draw header
        draw.text((padding, padding), "Model Comparison", fill='black', font=header_font)
        
        y_pos = padding + header_height
        
        # Draw ground truth if available
        if ground_truth is not None:
            gt_text = f"Ground Truth: {ground_truth[:60]}{'...' if len(ground_truth) > 60 else ''}"
            draw.text((padding, y_pos), gt_text, fill='green', font=font)
            y_pos += line_height
        
        # Draw predictions
        colors = ['blue', 'red', 'purple', 'orange', 'brown']
        for i, (model_name, prediction) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            pred_text = f"{model_name}: {prediction[:50]}{'...' if len(prediction) > 50 else ''}"
            draw.text((padding, y_pos), pred_text, fill=color, font=font)
            y_pos += line_height
        
        # Combine image and overlay
        combined = Image.new('RGB', (img_width, img_height + overlay_height))
        combined.paste(image, (0, 0))
        combined.paste(overlay, (0, img_height))
        
        if save_path:
            combined.save(save_path)
            logger.info(f"Model comparison saved to {save_path}")
        
        return combined
    
    def create_batch_visualization(
        self,
        images: List[Image.Image],
        predictions: List[str],
        ground_truths: Optional[List[str]] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Create a grid visualization of multiple inference results."""
        
        num_images = len(images)
        if num_images == 0:
            raise ValueError("No images provided")
        
        # Determine grid size
        if grid_size is None:
            cols = min(3, num_images)
            rows = (num_images + cols - 1) // cols
        else:
            rows, cols = grid_size
        
        # Standardize image sizes
        target_size = (300, 300)
        resized_images = []
        
        for i in range(min(num_images, rows * cols)):
            img = images[i]
            pred = predictions[i] if i < len(predictions) else ""
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            
            # Resize image while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create canvas and center image
            canvas = Image.new('RGB', target_size, color='white')
            img_w, img_h = img.size
            x = (target_size[0] - img_w) // 2
            y = (target_size[1] - img_h) // 2
            canvas.paste(img, (x, y))
            
            # Add text overlay for this image
            canvas = self._add_compact_text_overlay(canvas, pred, gt, i)
            resized_images.append(canvas)
        
        # Create grid
        cell_width, cell_height = target_size[0], target_size[1] + 60  # Extra height for text
        grid_width = cols * cell_width
        grid_height = rows * cell_height
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color='lightgray')
        
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height
            grid_image.paste(img, (x, y))
        
        if save_path:
            grid_image.save(save_path)
            logger.info(f"Batch visualization saved to {save_path}")
        
        return grid_image
    
    def _add_compact_text_overlay(
        self,
        image: Image.Image,
        prediction: str,
        ground_truth: Optional[str],
        index: int
    ) -> Image.Image:
        """Add compact text overlay for batch visualization."""
        
        img_width, img_height = image.size
        
        # Create text overlay
        overlay_height = 60
        overlay = Image.new('RGB', (img_width, overlay_height), color='white')
        draw = ImageDraw.Draw(overlay)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Draw index
        draw.text((5, 5), f"#{index + 1}", fill='black', font=font)
        
        # Draw prediction (truncated)
        pred_text = f"P: {prediction[:25]}{'...' if len(prediction) > 25 else ''}"
        draw.text((5, 20), pred_text, fill='blue', font=font)
        
        # Draw ground truth if available
        if ground_truth is not None:
            gt_text = f"GT: {ground_truth[:24]}{'...' if len(ground_truth) > 24 else ''}"
            draw.text((5, 35), gt_text, fill='green', font=font)
        
        # Combine
        combined = Image.new('RGB', (img_width, img_height + overlay_height))
        combined.paste(image, (0, 0))
        combined.paste(overlay, (0, img_height))
        
        return combined