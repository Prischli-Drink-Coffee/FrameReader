"""
Base classes for OCR system architecture.
Implements core interfaces for encoders, decoders and models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.image_size = config.get('image_size', (224, 224))
    
    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to hidden representations."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension of encoded features."""
        pass


class BaseDecoder(ABC, nn.Module):
    """Abstract base class for text decoders."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.vocab_size = config.get('vocab_size', 50265)
        self.max_length = config.get('max_length', 512)
    
    @abstractmethod
    def forward(
        self, 
        encoder_hidden_states: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode encoder states to text tokens."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output vocabulary dimension."""
        pass


class BaseOCRModel(ABC, nn.Module):
    """Abstract base class for complete OCR models."""
    
    def __init__(
        self, 
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if encoder.output_dim != decoder.hidden_size:
            self.projection = nn.Linear(encoder.output_dim, decoder.hidden_size)
        else:
            self.projection = nn.Identity()
    
    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through encoder and decoder."""
        pass
    
    @abstractmethod
    def generate(
        self,
        pixel_values: torch.Tensor,
        **kwargs
    ) -> Union[List[str], torch.Tensor]:
        """Generate text from images."""
        pass
    
    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        """Save model components to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.encoder.state_dict(), output_dir / "encoder.pt")
        torch.save(self.decoder.state_dict(), output_dir / "decoder.pt")
        
        if not isinstance(self.projection, nn.Identity):
            torch.save(self.projection.state_dict(), output_dir / "projection.pt")
        
        import json
        with open(output_dir / "model_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_dir: Union[str, Path], **kwargs) -> "BaseOCRModel":
        """Load model from directory."""
        pass
    
    def to_device(self, device: Union[str, torch.device]) -> "BaseOCRModel":
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        super().to(self.device)
        return self
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable model parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for training efficiency."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True