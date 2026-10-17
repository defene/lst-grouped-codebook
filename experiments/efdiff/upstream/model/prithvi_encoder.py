"""
Prithvi-EO-2.0-300M encoder wrapper for feature extraction.

The encoder loads pretrained Prithvi-EO-2.0-300M weights and extracts
patch-level embeddings from HLS (Harmonized Landsat Sentinel) imagery.

Model specs:
  - Architecture: ViT-L (24 layers, 16 heads)
  - Hidden dim: 1024
  - Patch size: [1, 16, 16] (temporal, height, width)
  - Input: 6 HLS bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
  - Image size: 224x224
  - Output: (B, 196, 1024) patch embeddings for 224x224 input
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger('base')

# Prithvi-EO-2.0 normalization constants (DN scale, i.e. reflectance * 10000)
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


class PrithviEncoder(nn.Module):
    """Frozen Prithvi-EO-2.0-300M encoder for HLS feature extraction.

    Loads the pretrained ViT backbone and extracts patch embeddings.
    All parameters are frozen (no gradient computation).

    Input:  HLS reflectance (B, 6, H, W) in [0, 1] range
    Output: Patch embeddings (B, N_patches, 1024) where N_patches = (H/16)*(W/16)
    """

    def __init__(
        self,
        model_name="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        embed_dim=1024,
        num_frames=1,
        freeze=True,
        input_is_reflectance=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.input_is_reflectance = input_is_reflectance

        # Register normalization constants as buffers (move to device automatically)
        self.register_buffer(
            'pixel_mean',
            torch.tensor(PRITHVI_MEAN).view(1, 6, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor(PRITHVI_STD).view(1, 6, 1, 1),
            persistent=False,
        )

        # Load the backbone
        self.backbone = self._load_backbone(model_name)

        if freeze:
            self._freeze()

    # Map from config model_name to terratorch short name
    _TERRATORCH_NAMES = {
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M": "prithvi_eo_v2_300",
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL": "prithvi_eo_v2_300_tl",
    }

    def _load_backbone(self, model_name):
        """Load Prithvi backbone via terratorch."""
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except ImportError:
            raise RuntimeError(
                "terratorch is required to load Prithvi models. "
                "Install it with: pip install terratorch")

        short_name = self._TERRATORCH_NAMES.get(model_name, model_name)
        backbone = BACKBONE_REGISTRY.build(short_name, pretrained=True)
        logger.info("Loaded Prithvi backbone via terratorch: {}".format(short_name))
        return backbone

    def _freeze(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("Prithvi encoder frozen ({:,d} params)".format(
            sum(p.numel() for p in self.backbone.parameters())))

    def train(self, mode=True):
        """Override train to keep backbone always in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def normalize(self, hls):
        """Normalize HLS input with Prithvi's mean/std.

        Args:
            hls: (B, 6, H, W) HLS reflectance in [0, 1]

        Returns:
            (B, 6, H, W) normalized for Prithvi
        """
        if self.input_is_reflectance:
            # Convert reflectance [0,1] to DN scale [0,10000]
            hls = hls * 10000.0
        return (hls - self.pixel_mean) / self.pixel_std

    def forward(self, hls):
        """Extract patch embeddings from HLS imagery.

        Args:
            hls: (B, 6, H, W) HLS reflectance data in [0, 1] range

        Returns:
            features: (B, N_patches, embed_dim) patch embeddings
                      For 224x224 input: (B, 196, 1024)
        """
        # Normalize input
        x = self.normalize(hls)

        # Prithvi expects (B, C, T, H, W) format
        # Add temporal dimension: (B, C, H, W) → (B, C, 1, H, W)
        x = x.unsqueeze(2)  # (B, 6, 1, H, W)

        with torch.no_grad():
            features = self._extract_features(x)

        return features

    def _extract_features(self, x):
        """Run backbone and extract features.

        Handles different backend output formats (terratorch, HF, timm).

        Args:
            x: (B, C, T, H, W) normalized input

        Returns:
            (B, N_patches, embed_dim) patch embeddings
        """
        B = x.shape[0]

        # Try standard forward
        out = self.backbone(x)

        # Handle different output formats
        if isinstance(out, (list, tuple)):
            # terratorch returns list of features from different layers
            # Use the last layer's features
            out = out[-1]

        if out.dim() == 3:
            # (B, N, D) — standard ViT output
            # Strip CLS token if present (197 -> 196 for 224x224 input)
            expected_patches = (x.shape[3] // 16) * (x.shape[4] // 16)
            if out.shape[1] == expected_patches + 1:
                out = out[:, 1:, :]  # remove CLS token
            return out
        elif out.dim() == 4:
            # (B, D, H, W) — CNN-style output, reshape to (B, N, D)
            B, D, H, W = out.shape
            return out.reshape(B, D, H * W).permute(0, 2, 1)
        elif out.dim() == 5:
            # (B, D, T, H, W) — 3D output, collapse temporal
            B, D, T, H, W = out.shape
            return out.reshape(B, D, T * H * W).permute(0, 2, 1)
        else:
            raise ValueError(
                "Unexpected Prithvi output shape: {}".format(out.shape))