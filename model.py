"""
GeoGuessr Hackathon - Model Module
CLIP ViT-B/32 backbone with regression heads for geolocation prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoLocationModel(nn.Module):
    """
    Geolocation prediction model using CLIP ViT-B/32 as backbone.

    Architecture:
        CLIP ViT-B/32 (visual encoder) -> 512-dim embedding
            -> MLP -> [pred_lat, pred_lon] (normalized to [-1, 1])
            -> MLP -> [pred_log_radius]    (log-scale radius in km)
    """

    def __init__(self, hidden_dim=256, dropout=0.3, num_classes=200):
        super().__init__()

        # Load CLIP ViT-B/32 visual encoder
        self.backbone, self.backbone_dim = self._load_clip_backbone()

        # Freeze backbone initially
        self._freeze_backbone()

        # Shared feature projection
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Coordinate regression head: predicts normalized [lat, lon]
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
            nn.Tanh(),  # Output in [-1, 1] matching normalized coords
        )

        # Radius prediction head: predicts log(radius_km)
        self.radius_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Classification head: predicts Geocell class
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_heads()

    def _load_clip_backbone(self):
        """Load CLIP ViT-B/32 visual encoder from torchvision or manual."""
        try:
            from transformers import CLIPVisionModel
            vit = CLIPVisionModel.from_pretrained("geolocal/StreetCLIP")
            print("[Model] Loaded geolocal/StreetCLIP from huggingface")
            return vit, vit.config.hidden_size
        except Exception as e:
            print(f"[Model] StreetCLIP failed to load: {e}")
            try:
                # Fallback: try CLIP from openai
                import clip
                model, _ = clip.load("ViT-B/32", device="cpu")
                visual = model.visual
                print("[Model] Loaded CLIP ViT-B/32 visual encoder")
                return visual, 512
            except Exception as e2:
                print(f"[Model] CLIP load failed: {e2}")
                # Final fallback: use ResNet50
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                resnet.fc = nn.Identity()
                print("[Model] Fallback to ResNet50 (backbone_dim=2048)")
                return resnet, 2048

    def _init_heads(self):
        """Initialize head weights."""
        for module in [self.feature_proj, self.coord_head, self.radius_head, self.class_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize radius head bias to log(500) ~ 6.2 (default 500km radius)
        with torch.no_grad():
            self.radius_head[-1].bias.fill_(math.log(500.0))

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen")

    def unfreeze_backbone_last_n(self, n=2):
        """Unfreeze the last N transformer blocks of the backbone."""
        if hasattr(self.backbone, 'encoder'):
            # HuggingFace CLIP Vision model
            blocks = list(self.backbone.encoder.layers)
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Unfreeze post-layer norm
            if hasattr(self.backbone, 'post_layernorm'):
                for param in self.backbone.post_layernorm.parameters():
                    param.requires_grad = True
            print(f"[Model] Unfroze last {n} transformer blocks")
        # For CLIP visual encoder
        elif hasattr(self.backbone, 'transformer'):
            blocks = list(self.backbone.transformer.resblocks)
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, 'ln_post'):
                for param in self.backbone.ln_post.parameters():
                    param.requires_grad = True
            print(f"[Model] Unfroze last {n} CLIP transformer blocks")
        else:
            # ResNet fallback - unfreeze layer4
            if hasattr(self.backbone, 'layer4'):
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
                print("[Model] Unfroze ResNet layer4")

    def unfreeze_all(self):
        """Unfreeze entire backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Full backbone unfrozen")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Image tensor [B, 3, 224, 224]

        Returns:
            coords: Predicted normalized [lat, lon] in [-1, 1], shape [B, 2]
            log_radius: Predicted log(radius_km), shape [B, 1]
            class_logits: Predicted logits for geocell classes, shape [B, num_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle HuggingFace CLIP output
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif features.dim() > 2:
            features = features.mean(dim=1)  # Global average pool if needed

        # Adjust feature dim if backbone changed (ResNet fallback)
        if features.shape[-1] != 512 and not hasattr(self, '_proj_adjusted'):
            self.feature_proj[1] = nn.Linear(features.shape[-1], 256).to(features.device)
            self._proj_adjusted = True

        # Project features
        projected = self.feature_proj(features)

        # Predict coordinates, radius, and class
        coords = self.coord_head(projected)       # [B, 2] in [-1, 1]
        log_radius = self.radius_head(projected)   # [B, 1]
        class_logits = self.class_head(projected)  # [B, num_classes]

        return coords, log_radius, class_logits

    def get_param_groups(self, lr_backbone=1e-5, lr_heads=1e-3):
        """Get parameter groups with different learning rates."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone'):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_heads},
        ]

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1, lat2, lon2: Coordinates in DEGREES

    Returns:
        Distance in kilometers
    """
    # Force float32 to prevent float16 mixed precision overflow/underflow
    lat1 = lat1.float()
    lon1 = lon1.float()
    lat2 = lat2.float()
    lon2 = lon2.float()

    R = 6371.0  # Earth radius in km

    lat1_r = torch.deg2rad(lat1)
    lat2_r = torch.deg2rad(lat2)
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)

    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, 0.0, 1.0)

    # Safe sqrt to prevent infinite gradients at a=0 (identical points) or a=1 (antipodal points)
    sqrt_a = torch.sqrt(torch.clamp(a, min=1e-9))
    sqrt_1_a = torch.sqrt(torch.clamp(1.0 - a, min=1e-9))

    c = 2 * torch.atan2(sqrt_a, sqrt_1_a)
    return R * c


class GeoLocationLoss(nn.Module):
    """
    Combined loss for geolocation prediction.

    Components:
        1. Haversine distance loss (primary)
        2. Coordinate MSE loss (auxiliary, for stability)
        3. Radius calibration loss (encourages well-calibrated radius)
    """

    def __init__(self, lambda_coord=1.0, lambda_radius=0.1, lambda_mse=0.5, lambda_cls=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_radius = lambda_radius
        self.lambda_mse = lambda_mse
        self.lambda_cls = lambda_cls

    def forward(self, pred_coords, pred_log_radius, pred_logits, target_coords, target_classes=None):
        """
        Args:
            pred_coords: [B, 2] predicted normalized [lat, lon] in [-1, 1]
            pred_log_radius: [B, 1] predicted log(radius_km)
            pred_logits: [B, num_classes] predicted geocell logits
            target_coords: [B, 2] target normalized [lat, lon] in [-1, 1]
            target_classes: [B] target geocell classes


        Returns:
            total_loss, loss_dict
        """
        # Denormalize coordinates for haversine
        pred_lat = pred_coords[:, 0] * 90.0
        pred_lon = pred_coords[:, 1] * 180.0
        true_lat = target_coords[:, 0] * 90.0
        true_lon = target_coords[:, 1] * 180.0

        # 1. Haversine distance loss
        distances = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
        # Use log(1 + distance) for smoother gradients at large distances
        haversine_loss = torch.mean(torch.log1p(distances))

        # 2. MSE loss on normalized coordinates (for training stability)
        mse_loss = F.mse_loss(pred_coords, target_coords)

        # 3. Radius calibration loss
        # Encourage log_radius to match log(actual_distance + epsilon)
        log_distances = torch.log(distances + 1.0)  # log(distance + 1)
        pred_log_r = pred_log_radius.squeeze(-1)
        # Loss: penalize if radius is too tight (doesn't cover true location)
        # Reward if radius is tight but covers the true location
        radius_loss = F.smooth_l1_loss(pred_log_r, log_distances.detach())

        # 4. Classification loss
        if target_classes is not None:
            cls_loss = F.cross_entropy(pred_logits, target_classes)
        else:
            cls_loss = torch.tensor(0.0, device=pred_coords.device)

        # Combined loss
        total_loss = (self.lambda_coord * haversine_loss +
                      self.lambda_mse * mse_loss +
                      self.lambda_radius * radius_loss +
                      self.lambda_cls * cls_loss)

        loss_dict = {
            'total': total_loss.item(),
            'haversine': haversine_loss.item(),
            'mse': mse_loss.item(),
            'radius': radius_loss.item(),
            'classification': cls_loss.item(),
            'median_dist_km': torch.median(distances).item(),
            'mean_dist_km': torch.mean(distances).item(),
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    # Quick test
    model = GeoLocationModel()
    trainable, total = model.count_parameters()
    print(f"\nParameters: {trainable:,} trainable / {total:,} total")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    coords, log_radius, logits = model(x)
    print(f"Coords shape: {coords.shape}, range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"Log radius shape: {log_radius.shape}, values: {log_radius.squeeze().tolist()}")
    print(f"Logits shape: {logits.shape}")

    # Test loss
    criterion = GeoLocationLoss()
    target_coords = torch.randn(4, 2).tanh()
    target_cls = torch.randint(0, 200, (4,))
    loss, loss_dict = criterion(coords, log_radius, logits, target_coords, target_cls)
    print(f"Loss: {loss_dict}")
