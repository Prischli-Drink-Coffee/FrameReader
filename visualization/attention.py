"""
Attention visualization for OCR models.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

import numpy as np
from PIL import Image
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available, attention visualization will be disabled")

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualizes attention mechanisms in OCR models."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("viridis")
    
    def visualize_encoder_attention(
        self,
        image: Image.Image,
        attention_weights: Union[torch.Tensor, np.ndarray],
        patch_size: Tuple[int, int] = (16, 16),
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Visualize encoder self-attention weights."""
        
        # Convert to numpy if tensor
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Handle multi-layer, multi-head attention
        if attention_weights.ndim == 4:  # [layers, heads, seq_len, seq_len]
            if layer_idx is not None:
                attention_weights = attention_weights[layer_idx]
            else:
                attention_weights = attention_weights.mean(axis=0)  # Average across layers
        
        if attention_weights.ndim == 3:  # [heads, seq_len, seq_len]
            if head_idx is not None:
                attention_weights = attention_weights[head_idx]
            else:
                attention_weights = attention_weights.mean(axis=0)  # Average across heads
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention_weights, cmap='viridis', interpolation='nearest')
        axes[1].set_title('Attention Weights')
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[1])
        
        # Attention overlay on image
        self._create_attention_overlay(axes[2], image, attention_weights, patch_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Encoder attention visualization saved to {save_path}")
        
        # Convert to PIL Image
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(plot_array)
        
        plt.close()
        return result_image
    
    def visualize_decoder_attention(
        self,
        image: Image.Image,
        tokens: List[str],
        cross_attention: Union[torch.Tensor, np.ndarray],
        self_attention: Optional[Union[torch.Tensor, np.ndarray]] = None,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Visualize decoder attention weights."""
        
        # Convert tensors to numpy
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.detach().cpu().numpy()
        if self_attention is not None and isinstance(self_attention, torch.Tensor):
            self_attention = self_attention.detach().cpu().numpy()
        
        # Handle multi-layer, multi-head attention
        cross_attention = self._process_attention_tensor(cross_attention, layer_idx, head_idx)
        if self_attention is not None:
            self_attention = self._process_attention_tensor(self_attention, layer_idx, head_idx)
        
        # Create visualization layout
        num_plots = 3 if self_attention is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Cross-attention heatmap
        if len(tokens) > cross_attention.shape[0]:
            tokens = tokens[:cross_attention.shape[0]]
        elif len(tokens) < cross_attention.shape[0]:
            tokens.extend([f"<pad{i}>" for i in range(cross_attention.shape[0] - len(tokens))])
        
        im1 = axes[1].imshow(cross_attention, cmap='Blues', aspect='auto')
        axes[1].set_title('Cross-Attention (Text → Image)')
        axes[1].set_xlabel('Image Patches')
        axes[1].set_ylabel('Text Tokens')
        axes[1].set_yticks(range(len(tokens)))
        axes[1].set_yticklabels(tokens, fontsize=8)
        plt.colorbar(im1, ax=axes[1])
        
        # Self-attention if available
        if self_attention is not None and num_plots == 3:
            im2 = axes[2].imshow(self_attention, cmap='Reds', aspect='auto')
            axes[2].set_title('Self-Attention (Text → Text)')
            axes[2].set_xlabel('Text Tokens')
            axes[2].set_ylabel('Text Tokens')
            axes[2].set_xticks(range(len(tokens)))
            axes[2].set_xticklabels(tokens, rotation=45, fontsize=8)
            axes[2].set_yticks(range(len(tokens)))
            axes[2].set_yticklabels(tokens, fontsize=8)
            plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decoder attention visualization saved to {save_path}")
        
        # Convert to PIL Image
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(plot_array)
        
        plt.close()
        return result_image
    
    def _process_attention_tensor(
        self,
        attention: np.ndarray,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> np.ndarray:
        """Process multi-dimensional attention tensor."""
        
        if attention.ndim == 4:  # [layers, heads, seq_len, seq_len]
            if layer_idx is not None:
                attention = attention[layer_idx]
            else:
                attention = attention.mean(axis=0)
        
        if attention.ndim == 3:  # [heads, seq_len, seq_len]
            if head_idx is not None:
                attention = attention[head_idx]
            else:
                attention = attention.mean(axis=0)
        
        return attention
    
    def _create_attention_overlay(
        self,
        ax: Any,  # plt.Axes when matplotlib is available
        image: Image.Image,
        attention_weights: np.ndarray,
        patch_size: Tuple[int, int]
    ) -> None:
        """Create attention overlay on the original image."""
        
        # Show image as background
        ax.imshow(image)
        ax.set_title('Attention Overlay')
        ax.axis('off')
        
        # Calculate patch grid
        img_width, img_height = image.size
        patch_w, patch_h = patch_size
        
        n_patches_w = img_width // patch_w
        n_patches_h = img_height // patch_h
        
        # Average attention across all queries (simplified visualization)
        attention_map = attention_weights.mean(axis=0)  # Average across query positions
        
        # Reshape to spatial grid if possible
        if len(attention_map) == n_patches_w * n_patches_h:
            attention_spatial = attention_map.reshape(n_patches_h, n_patches_w)
            
            # Create overlay
            overlay = np.zeros((img_height, img_width))
            
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    y_start = i * patch_h
                    y_end = min((i + 1) * patch_h, img_height)
                    x_start = j * patch_w
                    x_end = min((j + 1) * patch_w, img_width)
                    
                    overlay[y_start:y_end, x_start:x_end] = attention_spatial[i, j]
            
            # Apply overlay with transparency
            ax.imshow(overlay, alpha=0.6, cmap='hot', interpolation='bilinear')
    
    def visualize_attention_heads(
        self,
        image: Image.Image,
        attention_weights: Union[torch.Tensor, np.ndarray],
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        max_heads: int = 8,
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Visualize multiple attention heads simultaneously."""
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Extract specific layer
        if attention_weights.ndim == 4:  # [layers, heads, seq_len, seq_len]
            attention_weights = attention_weights[layer_idx]
        elif attention_weights.ndim != 3:
            raise ValueError(f"Expected 3D or 4D attention tensor, got {attention_weights.ndim}D")
        
        num_heads = min(attention_weights.shape[0], max_heads)
        
        # Create subplot grid
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if num_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()[:num_heads]
        else:
            axes = axes.flatten()[:num_heads]
        
        for head_idx in range(num_heads):
            head_attention = attention_weights[head_idx]
            
            # Create heatmap for this head
            im = axes[head_idx].imshow(head_attention, cmap='viridis', aspect='auto')
            axes[head_idx].set_title(f'Head {head_idx + 1}')
            
            # Add token labels if available
            if tokens is not None and len(tokens) == head_attention.shape[0]:
                axes[head_idx].set_yticks(range(len(tokens)))
                axes[head_idx].set_yticklabels(tokens, fontsize=6)
                axes[head_idx].set_xticks(range(len(tokens)))
                axes[head_idx].set_xticklabels(tokens, rotation=45, fontsize=6)
            
            plt.colorbar(im, ax=axes[head_idx])
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Attention Heads (Layer {layer_idx})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multi-head attention visualization saved to {save_path}")
        
        # Convert to PIL Image
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(plot_array)
        
        plt.close()
        return result_image
    
    def create_attention_rollout(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        image: Image.Image,
        tokens: List[str],
        save_path: Optional[Path] = None
    ) -> Image.Image:
        """Create attention rollout visualization showing information flow."""
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Compute attention rollout
        if attention_weights.ndim == 4:  # [layers, heads, seq_len, seq_len]
            # Average across heads first
            attention_weights = attention_weights.mean(axis=1)
            
            # Compute rollout across layers
            rollout = self._compute_attention_rollout(attention_weights)
        else:
            rollout = attention_weights
        
        # Visualize rollout
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention rollout heatmap
        im = axes[1].imshow(rollout, cmap='plasma', aspect='auto')
        axes[1].set_title('Attention Rollout')
        axes[1].set_xlabel('Source Position')
        axes[1].set_ylabel('Target Position')
        
        if tokens and len(tokens) == rollout.shape[0]:
            axes[1].set_yticks(range(len(tokens)))
            axes[1].set_yticklabels(tokens, fontsize=8)
        
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention rollout visualization saved to {save_path}")
        
        # Convert to PIL Image
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(plot_array)
        
        plt.close()
        return result_image
    
    def _compute_attention_rollout(self, attention_weights: np.ndarray) -> np.ndarray:
        """Compute attention rollout across multiple layers."""
        
        num_layers = attention_weights.shape[0]
        rollout = np.eye(attention_weights.shape[1])
        
        for layer_idx in range(num_layers):
            layer_attention = attention_weights[layer_idx]
            # Add residual connection
            layer_attention = layer_attention + np.eye(layer_attention.shape[0])
            # Normalize
            layer_attention = layer_attention / layer_attention.sum(axis=-1, keepdims=True)
            # Accumulate
            rollout = np.matmul(layer_attention, rollout)
        
        return rollout