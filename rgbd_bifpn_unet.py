import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF
from torch.serialization import add_safe_globals
from tqdm import tqdm

RGB_FILENAME = "rgb.png"
DEPTH_FILENAMES = {
    "depth_color": "depth_color.png",
    "depth_raw": "depth_raw.png",
}
# use the standard ImageNet normalization constants for RGB stream as mentioned in Table I.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOGGER = logging.getLogger(__name__)
add_safe_globals([Path])


@dataclass
class TrainConfig:
    data_root: Path
    depth_source: str
    train_csv: Path
    macro_csv: Optional[Path]
    output_dir: Path
    checkpoint_path: Optional[Path]
    epochs: int
    batch_size: int
    val_batch_size: int
    val_ratio: float
    seed: int
    num_workers: int
    use_amp: bool
    learning_rate: float
    weight_decay: float
    patience: int
    rgb_backbone: str
    depth_backbone: str
    fusion_channels: int
    bifpn_layers: int
    decoder_channels: Tuple[int, ...]
    predict_macros: bool
    # This defines the number of channels for widened depth input stack (e.g., color, raw, mask).
    depth_in_channels: int
    input_size: int
    max_grad_norm: float
    depth_threshold: float
    volume_scale: float
    num_bins: int
    log_interval: int
    # These parameters control self-supervised pretraining (SelfSup) stage.
    selfsup_epochs: int = 0
    selfsup_temperature: float = 0.2
    selfsup_reconstruction_weight: float = 0.5
    selfsup_depth_mask_ratio: float = 0.5
    selfsup_patch_size: int = 8
    selfsup_learning_rate: Optional[float] = None
    fuse_alpha: float = 1.0
    # This corresponds to preprocessing step of unifying image orientation for geometric consistency.
    canonical_orientation: str = "portrait"
    history_filename: str = "history.jsonl"
    save_interval: int = 50


@dataclass
class TargetNormalizer:
    # normalize the target calorie values for stable training.
    mean: float
    std: float

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / (self.std + 1e-6)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * (self.std + 1e-6) + self.mean


def seed_everything(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch for repeatable results.

    Args:
        seed: Random seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def robust_normalize_depth(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Normalize a depth map using median and IQR. Invalid pixels become 0.

    Args:
        depth: Depth image as float array.
        valid_mask: Boolean mask of valid pixels.

    Returns:
        Normalized depth map as float32.
    """
    valid_values = depth[valid_mask]
    if valid_values.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    median = np.median(valid_values)
    q75, q25 = np.percentile(valid_values, [75, 25])
    # the robust IQR normalization describe in Table I for depth data.
    iqr = max(q75 - q25, 1e-3)
    normalized = (depth - median) / iqr
    normalized[~valid_mask] = 0.0
    return normalized.astype(np.float32)


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load an RGB image and return float array in [0, 1].

    Args:
        path: File path to the RGB image.

    Returns:
        HxWx3 float32 RGB image scaled to [0, 1].
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load RGB image at {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image


def load_depth_color(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load colorized depth (3 channels) and make a valid mask.

    Args:
        path: File path to the depth_color image.

    Returns:
        Tuple of (C,H,W) normalized channels and (H,W) valid mask.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load depth color image at {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = np.any(image > 0.0, axis=2)
    channels = [
        robust_normalize_depth(image[..., idx], mask) for idx in range(image.shape[2])
    ]
    stacked = np.stack(channels, axis=0)
    return stacked, mask


def load_depth_raw(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw depth image, normalize, and build a valid mask.

    Args:
        path: File path to the raw depth image.

    Returns:
        (1,H,W) normalized depth, (H,W) valid mask, and raw depth (H,W).
    """
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to load raw depth image at {path}")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0
    else:
        depth = depth.astype(np.float32) / 255.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    mask = depth > 0.0
    # This raw, normalized depth is a key part of multi-channel depth input stack.
    normalized = robust_normalize_depth(depth, mask)[None, ...]
    return normalized, mask, depth


def compute_plane_volume(depth: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate volume above a best-fit plane inside the valid mask.

    Args:
        depth: Depth map (H,W) or (1,H,W).
        mask: Boolean valid mask (H,W).

    Returns:
        Mean positive residual to plane over valid area.
    """
    # for Depth-based Geometric Volume Prior calculation.
    if depth.ndim == 3:
        depth = depth[0]
    coords = np.argwhere(mask)
    if coords.shape[0] < 64:
        return 0.0
    sampled_idx = np.random.choice(
        coords.shape[0], size=min(5000, coords.shape[0]), replace=False
    )
    sample_coords = coords[sampled_idx]
    z = depth[sample_coords[:, 0], sample_coords[:, 1]]
    x = sample_coords[:, 1].astype(np.float32)
    y = sample_coords[:, 0].astype(np.float32)
    ones = np.ones_like(x)
    A = np.stack([x, y, ones], axis=1)
    try:
        # fit a supporting plane to the valid depth points.
        params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.clip(z - np.median(z), 0.0, None).sum())
    plane = (params[0] * x + params[1] * y + params[2])
    # then calculate the residual (difference) between the plane and the actual depth.
    residual = plane - z
    # integrate the positive residuals food above plane to get a volume proxy.
    residual = np.clip(residual, 0.0, None)
    volume = residual.sum()
    # This scalar value serves as a explicit geometric inductive bias for portion size.
    return float(volume / (mask.sum() + 1e-6))


def compute_statistics(
    dish_ids: Sequence[str],
    labels: Dict[str, float],
    data_root: Path,
    num_bins: int,
) -> Tuple[TargetNormalizer, np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Compute target stats, calorie bins, and depth volume features.

    Returns normalizer, calorie bin edges, per-ID volume (z-score), and
    volume mean/std.
    """
    calories = np.array([labels[_id] for _id in dish_ids], dtype=np.float32)
    target_normalizer = TargetNormalizer(
        mean=float(calories.mean()), std=float(calories.std(ddof=1) + 1e-6)
    )
    quantiles = np.linspace(0.0, 1.0, num=num_bins + 1)
    edges = np.quantile(calories, quantiles)
    calorie_bins = np.unique(edges).astype(np.float32)
    effective_bins = int(calorie_bins.shape[0] - 1)
    if effective_bins < num_bins:
        LOGGER.info(
            "Equal-frequency binning reduced num_bins from %d to %d due to duplicate quantiles.",
            num_bins, effective_bins
        )
    if effective_bins <= 0:
        calorie_bins = np.array([calories.min(), calories.max()], dtype=np.float32)

    volume_lookup_raw: Dict[str, float] = {}
    volumes: List[float] = []
    depth_root = data_root / "train" / "depth_raw"
    # precompute volume prior for all dishes before training starts.
    for dish_id in tqdm(dish_ids, desc="Precomputing depth volumes"):
        depth_path = depth_root / dish_id / DEPTH_FILENAMES["depth_raw"]
        try:
            _, mask, depth_raw = load_depth_raw(depth_path)
        except FileNotFoundError:
            depth_raw = np.zeros((1, 1), dtype=np.float32)
            mask = np.zeros_like(depth_raw, dtype=bool)
        # to compute_plane_volume gets the precomputed scalar.
        volume = compute_plane_volume(depth_raw, mask)
        volume_lookup_raw[dish_id] = volume
        volumes.append(volume)
    if volumes:
        volumes_np = np.array(volumes, dtype=np.float32)
        volume_mean = float(volumes_np.mean())
        volume_std = float(volumes_np.std(ddof=1) + 1e-6) if len(volumes) > 1 else float(
            volumes_np.std(ddof=0) + 1e-6
        )
    else:
        volume_mean = 0.0
        volume_std = 1.0
    # z-score normalize the volumes to use as a feature and target.
    volume_lookup = {
        dish_id: (volume_lookup_raw[dish_id] - volume_mean) / volume_std
        for dish_id in dish_ids
    }
    volume_stats = {"mean": volume_mean, "std": volume_std}
    return target_normalizer, calorie_bins, volume_lookup, volume_stats


def compute_calorie_stats_for_ids(
    dish_ids: Sequence[str],
    labels: Dict[str, float],
    num_bins: int,
) -> Tuple[TargetNormalizer, np.ndarray]:
    """
    Compute target normalizer and equal-frequency calorie bins.

    Args:
        dish_ids: IDs to use for stats.
        labels: Map from ID to calories.
        num_bins: Desired number of bins.

    Returns:
        Tuple of (normalizer, bin edges).
    """
    calories = np.array([labels[_id] for _id in dish_ids], dtype=np.float32)
    target_normalizer = TargetNormalizer(
        mean=float(calories.mean()),
        std=float(calories.std(ddof=1) + 1e-6),
    )
    # creates the equal-frequency bins for optional bucket head.
    quantiles = np.linspace(0.0, 1.0, num=num_bins + 1)
    edges = np.quantile(calories, quantiles)
    calorie_bins = np.unique(edges).astype(np.float32)
    effective_bins = int(calorie_bins.shape[0] - 1)
    if effective_bins < num_bins:
        LOGGER.info(
            "Equal-frequency binning reduced num_bins from %d to %d due to duplicate quantiles.",
            num_bins, effective_bins
        )
    if effective_bins <= 0:
        calorie_bins = np.array([calories.min(), calories.max()], dtype=np.float32)
    return target_normalizer, calorie_bins


def compute_bucket_weights(
    train_ids: Sequence[str],
    label_map: Dict[str, float],
    calorie_bins: np.ndarray,
) -> torch.Tensor:
    """
    Make class weights for calorie bins using inverse frequency.

    Args:
        train_ids: Training IDs.
        label_map: Map from ID to calories.
        calorie_bins: Calorie bin edges.

    Returns:
        1D tensor of weights with mean 1.0.
    """
    vals = np.array([label_map[i] for i in train_ids], dtype=np.float32)
    idx = np.clip(np.searchsorted(calorie_bins, vals, side="right") - 1, 0, len(calorie_bins) - 2)
    counts = np.bincount(idx, minlength=len(calorie_bins) - 1).astype(np.float32)
    # use inverse frequency weighting to help the bucket head handle imbalanced calorie distributions.
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_bin_midpoints_norm(calorie_bins: np.ndarray, target_normalizer: TargetNormalizer) -> torch.Tensor:
    """
    Get normalized midpoints of calorie bins as a 1xK tensor.

    Args:
        calorie_bins: Bin edges.
        target_normalizer: Normalizer for calories.
    """
    mids = 0.5 * (calorie_bins[:-1] + calorie_bins[1:])
    mids_t = torch.tensor(mids, dtype=torch.float32)
    mids_norm = (mids_t - target_normalizer.mean) / (target_normalizer.std + 1e-6)
    return mids_norm.view(1, -1)


def setup_logging(output_dir: Path) -> None:
    """
    Set up console and file logging under the output directory.

    Args:
        output_dir: Folder to write logs into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    LOGGER.info("Logging initialised. Writing to %s", log_path)


def append_history_entry(output_dir: Path, filename: str, entry: Dict[str, Any]) -> None:
    """
    Append a single history record to JSONL and a wide CSV for plotting.

    JSONL keeps the raw entry, while CSV flattens metrics so each row has
    columns: epoch, phase, and metric keys as columns.
    """
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    try:
        csv_name = Path(filename).with_suffix(".csv").name
        csv_path = output_dir / csv_name
        # Build one-row dataframe for the entry
        base = {
            "epoch": entry.get("epoch"),
            "phase": entry.get("phase"),
        }
        metrics = entry.get("metrics", {}) or {}
        # Only keep scalar-like values in CSV
        row = {**base}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                row[k] = float(v)
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            try:
                df_old = pd.read_csv(csv_path)
                # Ensure union of columns; fill missing with NaN
                for col in df_new.columns:
                    if col not in df_old.columns:
                        df_old[col] = np.nan
                for col in df_old.columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan
                df_all = pd.concat([df_old, df_new[df_old.columns]], ignore_index=True)
            except Exception:
                # If the existing CSV is malformed, fall back to overwrite
                df_all = df_new
        else:
            df_all = df_new
        # Reorder columns: epoch, phase, then sorted metric keys
        cols = [c for c in df_all.columns if c not in ("epoch", "phase")]
        cols_sorted = sorted(cols)
        ordered_cols = ["epoch", "phase", *cols_sorted]
        df_all.to_csv(csv_path, index=False, columns=ordered_cols)
    except Exception as e:
        # CSV export should not break training; log and continue
        LOGGER.warning("Failed to update history CSV: %s", e)


def create_grad_scaler(use_amp: bool) -> amp.GradScaler:
    """
    Create an AMP GradScaler if CUDA AMP is available and enabled.

    Args:
        use_amp: If false, return a disabled scaler.

    Returns:
        A GradScaler instance.
    """
    if not torch.cuda.is_available() or not use_amp:
        return amp.GradScaler(enabled=False)
    try:
        from torch import amp as torch_amp  # type: ignore

        grad_scaler_cls = getattr(torch_amp, "GradScaler")
        return grad_scaler_cls(device_type="cuda", enabled=True)
    except (ImportError, AttributeError, TypeError):
        return amp.GradScaler(enabled=True)


def apply_depth_patch_mask(
    depth: torch.Tensor,
    ratio: float,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask patches of the depth feature tensor (channel-first).

    Returns (masked_depth, keep_mask) where keep_mask is a binary map in [0,1].

    Args:
        depth: Input depth feature map (B,C,H,W).
        ratio: Fraction of patches to drop.
        patch_size: Patch size in pixels.
    """
    # implements the "Depth Masking" from Self-Supervised Training (Fig. 1).
    if ratio <= 0.0:
        mask = depth.new_ones((depth.size(0), 1, depth.size(2), depth.size(3)))
        return depth, mask
    b, _, h, w = depth.shape
    patch = max(1, patch_size)
    grid_h = h // patch
    grid_w = w // patch
    if grid_h == 0 or grid_w == 0:
        mask = depth.new_ones((b, 1, h, w))
        return depth, mask
    total = grid_h * grid_w
    keep = max(1, int(round((1.0 - ratio) * total)))
    mask_flat = depth.new_zeros((b, total))
    for idx in range(b):
        keep_indices = torch.randperm(total, device=depth.device)[:keep]
        mask_flat[idx, keep_indices] = 1.0
    mask = mask_flat.view(b, 1, grid_h, grid_w)
    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    # return the depth multiplied by the mask, forcing the model to learn from incomplete geometric cues.
    return depth * mask, mask


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConv2d(nn.Module):
    # a building block for the BiFPN module, which uses separable convolutions.
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ResNetEncoder(nn.Module):
    def __init__(self, name: str = "resnet34", in_channels: int = 3) -> None:
        super().__init__()
        if name != "resnet34":
            raise ValueError(f"Unsupported ResNet encoder: {name}. Only resnet34 is supported.")
        # As stated in report, use ResNet-34 for both streams.
        backbone_fn = getattr(models, name)
        backbone = backbone_fn(weights=None)
        if in_channels != 3:
            # modify the first layer to accept multi-channel depth stack.
            self._replace_first_conv(backbone, in_channels)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # extract multi-scale features, which will be fed into the BiFPN.
        self.out_channels = [64, 128, 256, 512]

    @staticmethod
    def _replace_first_conv(backbone: nn.Module, in_channels: int) -> None:
        # This helper function adapts the ResNet's first convolution for widened depth input.
        old_conv: nn.Conv2d = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                repeat = int(torch.ceil(torch.tensor(in_channels / old_conv.weight.shape[1])).item())
                expanded = old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(expanded / float(repeat))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # return features from 4 different scales for the BiFPN fusion.
        return [c2, c3, c4, c5]


class BiFPNBlock(nn.Module):
    # This module implements the BiFPN-style fusion from report's architecture.
    def __init__(self, channels: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = epsilon
        # These are the learnable, non-negative weights for bidirectional paths.
        self.w_top = nn.Parameter(torch.ones(3, 2))
        self.w_bottom = nn.Parameter(torch.ones(3, 3))
        self.top_convs = nn.ModuleList([SeparableConv2d(channels, channels) for _ in range(3)])
        self.out_convs = nn.ModuleList([SeparableConv2d(channels, channels) for _ in range(4)])

    def _normalize(self, weights: torch.Tensor) -> torch.Tensor:
        # use a ReLU + normalization to ensure weights are non-negative and sum to 1.
        weights = F.relu(weights)
        return weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != 4:
            raise ValueError("BiFPNBlock expects exactly four feature levels.")
        p2, p3, p4, p5 = features
        w_top = self._normalize(self.w_top)
        # This section implements the top-down pathway of the BiFPN.
        p5_td = p5
        p4_td = self.top_convs[0](
            w_top[0, 0] * p4 + w_top[0, 1] * F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest")
        )
        p3_td = self.top_convs[1](
            w_top[1, 0] * p3 + w_top[1, 1] * F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        )
        p2_td = self.top_convs[2](
            w_top[2, 0] * p2 + w_top[2, 1] * F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")
        )
        w_bottom = self._normalize(self.w_bottom)
        # This section implements the bottom-up pathway, completing the bidirectional fusion.
        p2_out = self.out_convs[0](p2_td)
        p3_in = (
            w_bottom[0, 0] * p3
            + w_bottom[0, 1] * p3_td
            + w_bottom[0, 2] * F.max_pool2d(p2_out, kernel_size=2, stride=2)
        )
        p3_out = self.out_convs[1](p3_in)
        p4_in = (
            w_bottom[1, 0] * p4
            + w_bottom[1, 1] * p4_td
            + w_bottom[1, 2] * F.max_pool2d(p3_out, kernel_size=2, stride=2)
        )
        p4_out = self.out_convs[2](p4_in)
        p5_in = (
            w_bottom[2, 0] * p5
            + w_bottom[2, 1] * p5_td
            + w_bottom[2, 2] * F.max_pool2d(p4_out, kernel_size=2, stride=2)
        )
        p5_out = self.out_convs[3](p5_in)
        return [p2_out, p3_out, p4_out, p5_out]


class DecoderBlock(nn.Module):
    # a standard convolutional block for U-Net style decoder.
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # It upsamples the feature map and concatenates it with a skip connection.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class MultiTaskDecoder(nn.Module):
    # This module implements Multi-Head Decoder (see Fig. 1) for multi-task learning.
    def __init__(
        self,
        feature_channels: int,
        decoder_channels: Sequence[int],
        predict_macros: bool = True,
        macro_count: int = 3,
        bucket_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(decoder_channels) == 0:
            raise ValueError("decoder_channels must contain at least one stage.")
        # These are the U-Net decoder blocks.
        self.blocks = nn.ModuleList()
        in_channels = feature_channels
        for out_channels in decoder_channels:
            self.blocks.append(DecoderBlock(in_channels, feature_channels, out_channels))
            in_channels = out_channels
        # Head (iii): Segmentation head to provide food mask.
        self.segmentation_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Head (ii): Calorie density map head.
        self.calorie_head = nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        # Head (i): The primary calorie regression head (GAP to MLP).
        self.calorie_total_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, 1),
        )
        # Head (iv): Volume proxy head, regularized by depth-based prior.
        self.volume_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, 1),
        )
        # the bucket head for ordinal classification.
        self.bucket_count = bucket_count
        if bucket_count is not None and bucket_count > 1:
            self.calorie_bucket_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels),
                nn.SiLU(inplace=True),
                nn.Linear(in_channels, bucket_count),
            )
        else:
            self.calorie_bucket_head = None
        self.predict_macros = predict_macros
        if predict_macros:
            self.macros_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels),
                nn.SiLU(inplace=True),
                nn.Linear(in_channels, macro_count),
            )
        else:
            self.macros_head = None

    def forward(self, features: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(features) != len(self.blocks) + 1:
            raise ValueError("Number of features must equal decoder blocks + 1.")
        # pass the fused features through the U-Net decoder blocks.
        x = features[-1]
        for idx, block in enumerate(self.blocks):
            skip = features[-(idx + 2)]
            x = block(x, skip)
        # then get predictions from all attached heads.
        segmentation_logits = self.segmentation_head(x)
        calorie_map = self.calorie_head(x)
        calorie_total = self.calorie_total_head(x)
        volume = self.volume_head(x)
        outputs: Dict[str, torch.Tensor] = {
            "segmentation": segmentation_logits,
            "calorie_map": calorie_map,
            "total_calories": calorie_total,
            "volume": volume,
            "decoder_features": x,
        }
        if self.calorie_bucket_head is not None:
            outputs["calorie_logits"] = self.calorie_bucket_head(x)
        if self.predict_macros and self.macros_head is not None:
            outputs["macros"] = self.macros_head(x)
        return outputs


class RGBDVersion1(nn.Module):
    # main model, RGB-D+BiFPN+U-Net, which combines all components from Fig. 1.
    def __init__(
        self,
        rgb_backbone: str = "resnet34",
        depth_backbone: str = "resnet34",
        depth_in_channels: int = 1,
        fusion_channels: int = 256,
        num_bifpn_layers: int = 2,
        decoder_channels: Sequence[int] = (256, 192, 128),
        predict_macros: bool = True,
        macro_count: int = 3,
        bucket_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        # define the dual-stream encoders.
        self.rgb_encoder = ResNetEncoder(rgb_backbone, in_channels=3)
        self.depth_encoder = ResNetEncoder(depth_backbone, in_channels=depth_in_channels)
        self.num_levels = 4
        self.fusion_channels = fusion_channels
        # These projection layers bring RGB and Depth features to a common channel dimension.
        self.rgb_projections = nn.ModuleList(
            [nn.Conv2d(in_ch, fusion_channels, kernel_size=1, bias=False) for in_ch in self.rgb_encoder.out_channels]
        )
        self.depth_projections = nn.ModuleList(
            [nn.Conv2d(in_ch, fusion_channels, kernel_size=1, bias=False) for in_ch in self.depth_encoder.out_channels]
        )
        self.proj_norms = nn.ModuleList(
            [nn.BatchNorm2d(fusion_channels) for _ in range(self.num_levels * 2)]
        )
        # These are the learnable weights for the Projection & Local Fusion step in Fig. 1.
        self.level_fusion_weights = nn.Parameter(torch.ones(self.num_levels, 2))
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(fusion_channels * 2, fusion_channels),
                    ConvBNAct(fusion_channels, fusion_channels),
                )
                for _ in range(self.num_levels)
            ]
        )
        # stack multiple BiFPN blocks for multi-scale fusion.
        self.bifpn = nn.ModuleList([BiFPNBlock(fusion_channels) for _ in range(num_bifpn_layers)])
        if len(decoder_channels) != self.num_levels - 1:
            raise ValueError("decoder_channels length must be num_levels - 1.")
        # Finally, attach multi-task U-Net decoder.
        self.decoder = MultiTaskDecoder(
            feature_channels=fusion_channels,
            decoder_channels=decoder_channels,
            predict_macros=predict_macros,
            macro_count=macro_count,
            bucket_count=bucket_count,
        )

    def _normalize_level_weights(self) -> torch.Tensor:
        weights = F.relu(self.level_fusion_weights)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-4)

    def _project_features(
        self,
        features: Sequence[torch.Tensor],
        projections: nn.ModuleList,
        norms: Sequence[nn.Module],
    ) -> List[torch.Tensor]:
        projected: List[torch.Tensor] = []
        for idx, feat in enumerate(features):
            proj = projections[idx](feat)
            proj = norms[idx](proj)
            projected.append(proj)
        return projected

    def _encode_and_fuse(self, rgb: torch.Tensor, depth: torch.Tensor) -> List[torch.Tensor]:
        # pass inputs through their respective ResNet-34 backbones.
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)
        # project the features to a common dimension.
        rgb_proj = self._project_features(rgb_feats, self.rgb_projections, self.proj_norms[: self.num_levels])
        depth_proj = self._project_features(
            depth_feats, self.depth_projections, self.proj_norms[self.num_levels :]
        )
        level_weights = self._normalize_level_weights()
        fused_levels: List[torch.Tensor] = []
        # the Projection & Local Fusion step.
        for idx in range(self.num_levels):
            rgb_feat = rgb_proj[idx]
            depth_feat = depth_proj[idx]
            # combine RGB and Depth features using learnable weights.
            base = level_weights[idx, 0] * rgb_feat + level_weights[idx, 1] * depth_feat
            stacked = torch.cat([rgb_feat, depth_feat], dim=1)
            fusion = self.fusion_convs[idx](stacked)
            fused_levels.append(fusion + base)
        # The locally fused features are then passed to the BiFPN for global multi-scale fusion.
        for block in self.bifpn:
            fused_levels = block(fused_levels)
        return fused_levels

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth tensors must share the same spatial dimensions.")
        # 1. Encode and fuse the dual-stream inputs.
        fused_features = self._encode_and_fuse(rgb, depth)
        # 2. Pass the fused features to the multi-head decoder.
        decoder_outputs = self.decoder(fused_features)
        input_size = rgb.shape[-2:]
        # 3. Upsample pixel-wise predictions (segmentation, density map) to the input size.
        segmentation = F.interpolate(
            decoder_outputs["segmentation"], size=input_size, mode="bilinear", align_corners=False
        )
        calorie_map = F.interpolate(
            decoder_outputs["calorie_map"], size=input_size, mode="bilinear", align_corners=False
        )
        outputs: Dict[str, torch.Tensor] = {
            "segmentation": segmentation,
            "calorie_map": calorie_map,
            "total_calories": decoder_outputs["total_calories"],
            "volume": decoder_outputs["volume"],
        }
        if "calorie_logits" in decoder_outputs:
            outputs["calorie_logits"] = decoder_outputs["calorie_logits"]
        if "macros" in decoder_outputs:
            outputs["macros"] = decoder_outputs["macros"]
        outputs["decoder_features"] = decoder_outputs["decoder_features"]
        return outputs


class MultitaskLoss(nn.Module):
    # This class defines the combined loss for multi-task learning setup.
    def __init__(
        self,
        segmentation_loss: Optional[nn.Module] = None,
        calorie_map_loss: Optional[nn.Module] = None,
        calorie_total_loss: Optional[nn.Module] = None,
        volume_loss: Optional[nn.Module] = None,
        macro_loss: Optional[nn.Module] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        calorie_bins: Optional[np.ndarray] = None,
        target_normalizer: Optional[TargetNormalizer] = None,
        bucket_weights: Optional[torch.Tensor] = None,
        ce_label_smoothing: float = 0.02,
        ce_temperature: float = 1.0,
        extreme_margin: float = 0.15,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss or nn.BCEWithLogitsLoss()
        self.calorie_map_loss = calorie_map_loss or nn.L1Loss()
        # As described in report, initialize the main loss to SmoothL1.
        self.calorie_total_loss = calorie_total_loss or nn.SmoothL1Loss(beta=1.0)
        self.volume_loss = volume_loss or nn.L1Loss()
        self.macro_loss = macro_loss or nn.SmoothL1Loss()

        self.bucket_loss = nn.CrossEntropyLoss(weight=bucket_weights, label_smoothing=ce_label_smoothing)
        self.ce_temperature = ce_temperature

        self._use_kcal_space_for_total = False
        self._smoothl1_beta = 1.0
        self.target_normalizer = target_normalizer
        self._target_normalizer = target_normalizer

        # These are the default weights for each head in multi-task objective.
        default_weights = {
            "segmentation": 0.05,
            "calorie_map": 0.0,
            "calorie_total": 1.0,
            "volume": 0.0,
            "macros": 0.0,
            "calorie_bucket": 0.50,
            "calorie_expectation": 0.00,
            "calorie_from_map": 0.10,
            "bucket_index": 0.10,
            "bucket_distance": 0.10,
            "bucket_prior": 0.05,
        }
        self.loss_weights = loss_weights or default_weights
        self.bin_midpoints_norm: Optional[torch.Tensor] = None
        self.tn_mean = float(target_normalizer.mean) if target_normalizer else 0.0
        self.tn_std = float(target_normalizer.std) if target_normalizer else 1.0
        self.extreme_margin = float(extreme_margin)

        if calorie_bins is not None and target_normalizer is not None:
            self.set_calorie_bins(calorie_bins, target_normalizer)

    def set_target_normalizer(self, tn: "TargetNormalizer") -> None:
        self.target_normalizer = tn
        self._target_normalizer = tn
        self.tn_mean = float(tn.mean)
        self.tn_std = float(tn.std)

    def _set_total_loss_impl(self, kind: str, beta: float = 1.0) -> None:
        kind = kind.lower()
        if kind in ("smoothl1", "huber"):
            self._smoothl1_beta = float(beta)
            self.calorie_total_loss = nn.SmoothL1Loss(beta=self._smoothl1_beta)
        elif kind in ("mse", "l2"):
            # switch to MSE loss after 50 epochs.
            self.calorie_total_loss = nn.MSELoss()
        elif kind in ("l1", "mae"):
            self.calorie_total_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported total loss kind: {kind}")

    def set_calorie_total_loss(self, kind: str = "smoothl1", *, beta: float = 1.0, use_kcal_space: bool = False) -> None:
        # allows us to change the main loss function during training.
        self._set_total_loss_impl(kind, beta=beta)
        self._use_kcal_space_for_total = bool(use_kcal_space)

    def set_calorie_bins(self, calorie_bins: np.ndarray, target_normalizer: TargetNormalizer) -> None:
        mids = 0.5 * (calorie_bins[:-1] + calorie_bins[1:])
        mids_t = torch.tensor(mids, dtype=torch.float32)
        mids_norm = (mids_t - target_normalizer.mean) / (target_normalizer.std + 1e-6)
        self.bin_midpoints_norm = mids_norm.view(1, -1)
        self.tn_mean = float(target_normalizer.mean)
        self.tn_std = float(target_normalizer.std)
        self._target_normalizer = target_normalizer

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = None
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            raise ValueError("Predictions dictionary does not contain any tensors.")
        total_loss = torch.zeros((), device=device)
        metrics: Dict[str, float] = {}

        # Loss for segmentation head.
        if "segmentation" in targets and "segmentation" in predictions:
            seg_loss = self.segmentation_loss(predictions["segmentation"], targets["segmentation"])
            weight = self.loss_weights.get("segmentation", 1.0)
            total_loss = total_loss + weight * seg_loss
            metrics["segmentation"] = float(seg_loss.detach().cpu())

        # Loss for calorie density map head.
        if "calorie_map" in targets and "calorie_map" in predictions:
            cal_map_loss = self.calorie_map_loss(predictions["calorie_map"], targets["calorie_map"])
            weight = self.loss_weights.get("calorie_map", 1.0)
            total_loss = total_loss + weight * cal_map_loss
            metrics["calorie_map"] = float(cal_map_loss.detach().cpu())
            try:
                # This implements the mask density consistency from Fig. 1.
                # check that the sum of the density map is consistent with the total calorie prediction.
                pred_density = F.relu(predictions["calorie_map"])  # Density must be non-negative
                if "segmentation" in predictions:
                    seg_prob = torch.sigmoid(predictions["segmentation"])
                else:
                    seg_prob = torch.ones_like(pred_density)
                pred_total_from_map = (pred_density * seg_prob).sum(dim=(1, 2, 3), keepdim=True)
                tn_mean = torch.as_tensor(self.tn_mean, device=pred_total_from_map.device, dtype=pred_total_from_map.dtype)
                tn_std = torch.as_tensor(self.tn_std, device=pred_total_from_map.device, dtype=pred_total_from_map.dtype)
                pred_total_from_map_norm = (pred_total_from_map - tn_mean) / (tn_std + 1e-6)
                cal_target_norm = targets.get("total_calories_norm", targets.get("total_calories"))
                if cal_target_norm is not None:
                    if cal_target_norm.shape != pred_total_from_map_norm.shape:
                        cal_target_norm = cal_target_norm.view_as(pred_total_from_map_norm)
                    from_map_l1 = F.l1_loss(pred_total_from_map_norm, cal_target_norm)
                    w = self.loss_weights.get("calorie_from_map", 0.0)
                    total_loss = total_loss + w * from_map_l1
                    metrics["calorie_from_map_l1"] = float(from_map_l1.detach().cpu())
            except Exception:
                pass

        # the main loss for the primary total calorie regression head.
        if "total_calories" in predictions:
            if self._use_kcal_space_for_total and self._target_normalizer is not None:
                # After switching to MSE, compute the loss in the original kcal space.
                pred_total = predictions["total_calories"]
                tn = self._target_normalizer
                pred_kcal = pred_total * (tn.std + 1e-6) + tn.mean
                calorie_target = targets.get("total_calories")
                if calorie_target is None:
                    cal_norm = targets.get("total_calories_norm")
                    if cal_norm is not None:
                        calorie_target = cal_norm * (tn.std + 1e-6) + tn.mean
                if calorie_target is not None:
                    if calorie_target.shape != pred_kcal.shape:
                        calorie_target = calorie_target.view_as(pred_kcal)
                    cal_total_loss = self.calorie_total_loss(pred_kcal, calorie_target)
                else:
                    calorie_target = targets.get("total_calories_norm")
                    if calorie_target is not None and calorie_target.shape != pred_total.shape:
                        calorie_target = calorie_target.view_as(pred_total)
                    cal_total_loss = self.calorie_total_loss(pred_total, calorie_target)
            else:
                # Initially (with SmoothL1), compute loss in the normalized space.
                calorie_target = targets.get("total_calories_norm", targets.get("total_calories"))
                pred_total = predictions["total_calories"]
                if calorie_target is not None and calorie_target.shape != pred_total.shape:
                    calorie_target = calorie_target.view_as(pred_total)
                cal_total_loss = self.calorie_total_loss(pred_total, calorie_target)
            weight = self.loss_weights.get("calorie_total", 0.5)
            total_loss = total_loss + weight * cal_total_loss
            metrics["total_calories"] = float(cal_total_loss.detach().cpu())

        # Loss for the volume proxy head, regularized by precomputed volume prior.
        if "volume" in targets and "volume" in predictions:
            pred_volume = predictions["volume"]
            target_volume = targets["volume"]
            if target_volume.shape != pred_volume.shape:
                target_volume = target_volume.view_as(pred_volume)
            volume_loss = self.volume_loss(pred_volume, target_volume)
            weight = self.loss_weights.get("volume", 0.5)
            total_loss = total_loss + weight * volume_loss
            metrics["volume"] = float(volume_loss.detach().cpu())

        # Loss for optional macro-nutrient head.
        if "macros" in targets and "macros" in predictions:
            macro_loss = self.macro_loss(predictions["macros"], targets["macros"])
            weight = self.loss_weights.get("macros", 0.25)
            total_loss = total_loss + weight * macro_loss
            metrics["macros"] = float(macro_loss.detach().cpu())

        # Losses for the optional calorie bucket head.
        if "bucket" in targets and "calorie_logits" in predictions:
            logits = predictions["calorie_logits"] / self.ce_temperature
            bucket = targets["bucket"].view(-1)

            if self.extreme_margin > 0.0:
                num_classes = logits.size(-1)
                is_extreme = (bucket.eq(0) | bucket.eq(num_classes - 1)).float().view(-1, 1)
                is_extreme = is_extreme.to(logits.dtype)
                add = torch.zeros_like(logits)
                src = (is_extreme * self.extreme_margin).to(logits.dtype)
                add.scatter_(1, bucket.view(-1, 1), src)
                logits = logits + add

            ce_loss = self.bucket_loss(logits, bucket)
            weight = self.loss_weights.get("calorie_bucket", 0.50)
            total_loss = total_loss + weight * ce_loss
            metrics["bucket_ce"] = float(ce_loss.detach().cpu())
            with torch.no_grad():
                pred_cls = torch.argmax(logits, dim=-1)
                acc = (pred_cls == bucket).float().mean()
                metrics["bucket_acc"] = float(acc.detach().cpu())
                probs_dbg = torch.softmax(logits, dim=-1)
                ent = (-probs_dbg.clamp_min(1e-8).log() * probs_dbg).sum(dim=-1).mean()
                metrics["bucket_entropy"] = float(ent.detach().cpu())

            probs = torch.softmax(logits, dim=-1)
            # can also add a loss to make the expected value of the buckets match the regression target.
            if self.bin_midpoints_norm is not None:
                mids = self.bin_midpoints_norm.to(probs.device)
                exp_norm = (probs * mids).sum(dim=-1, keepdim=True)
                calorie_target = targets.get("total_calories_norm", targets.get("total_calories"))
                if calorie_target is not None:
                    if calorie_target.shape != exp_norm.shape:
                        calorie_target = calorie_target.view_as(exp_norm)
                    exp_l1 = F.l1_loss(exp_norm, calorie_target)
                    weight = self.loss_weights.get("calorie_expectation", 0.0)
                    total_loss = total_loss + weight * exp_l1
                    metrics["bucket_expectation_l1"] = float(exp_l1.detach().cpu())

            idx = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype).view(1, -1)
            e_idx = (probs * idx).sum(dim=-1, keepdim=True)
            bucket_f = bucket.float().view_as(e_idx)
            idx_l1 = F.l1_loss(e_idx, bucket_f)
            w_idx = self.loss_weights.get("bucket_index", 0.10)
            total_loss = total_loss + w_idx * idx_l1
            metrics["bucket_index_l1"] = float(idx_l1.detach().cpu())

            dist = (idx - bucket_f).abs()
            dist_l1 = (probs * dist).sum(dim=-1).mean()
            w_dist = self.loss_weights.get("bucket_distance", 0.10)
            total_loss = total_loss + w_dist * dist_l1
            metrics["bucket_distance_l1"] = float(dist_l1.detach().cpu())

            with torch.no_grad():
                num_classes = probs.size(-1)
                tgt_hist = torch.zeros(num_classes, device=probs.device, dtype=probs.dtype)
                tgt_hist.scatter_add_(0, bucket, torch.ones_like(bucket, dtype=probs.dtype))
                tgt_hist = tgt_hist / tgt_hist.sum().clamp_min(1e-6)
            pred_hist = probs.mean(dim=0)
            kl = (tgt_hist.clamp_min(1e-8) * (tgt_hist.clamp_min(1e-8).log() - pred_hist.clamp_min(1e-8).log())).sum()
            w_prior = self.loss_weights.get("bucket_prior", 0.05)
            total_loss = total_loss + w_prior * kl
            metrics["bucket_prior_kl"] = float(kl.detach().cpu())

        return total_loss, metrics


class Nutrition5KMultitaskDataset(Dataset):
    def __init__(
        self,
        dish_ids: Sequence[str],
        data_root: Path,
        input_size: int,
        train: bool,
        target_normalizer: TargetNormalizer,
        calorie_bins: np.ndarray,
        volume_lookup: Dict[str, float],
        volume_stats: Dict[str, float],
        labels: Optional[Dict[str, float]] = None,
        macros: Optional[Dict[str, Tuple[float, ...]]] = None,
        depth_threshold: float = 0.5,
        canonical_orientation: str = "portrait",
    ) -> None:
        label_lookup = labels or {}
        requested_ids = list(dish_ids)
        valid_ids: List[str] = []
        missing = 0
        for dish_id in requested_ids:
            split = "train" if dish_id in label_lookup else "test"
            rgb_path = data_root / split / "color" / dish_id / RGB_FILENAME
            depth_color_path = data_root / split / "depth_color" / dish_id / DEPTH_FILENAMES["depth_color"]
            depth_raw_path = data_root / split / "depth_raw" / dish_id / DEPTH_FILENAMES["depth_raw"]
            # ensure all modalities (RGB, depth_color, depth_raw) are present.
            if not rgb_path.exists():
                LOGGER.warning("Skipping %s: missing RGB image at %s", dish_id, rgb_path)
                missing += 1
                continue
            if not depth_color_path.exists():
                LOGGER.warning("Skipping %s: missing depth_color image at %s", dish_id, depth_color_path)
                missing += 1
                continue
            if not depth_raw_path.exists():
                LOGGER.warning("Skipping %s: missing depth_raw image at %s", dish_id, depth_raw_path)
                missing += 1
                continue
            valid_ids.append(dish_id)
        if missing > 0:
            LOGGER.warning("Filtered out %d samples without complete modality data.", missing)
        if not valid_ids:
            raise ValueError("No valid samples found after filtering missing modality data.")
        self.dish_ids = valid_ids
        self.data_root = data_root
        self.input_size = input_size
        self.train = train
        self.labels = label_lookup
        self.macros = macros
        self.target_normalizer = target_normalizer
        self.calorie_bins = calorie_bins
        # pass in the precomputed volume priors.
        self.volume_lookup = volume_lookup
        self.volume_stats = volume_stats
        self.rgb_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        self.depth_threshold = depth_threshold
        self.canonical_orientation = canonical_orientation

    def __len__(self) -> int:
        return len(self.dish_ids)

    def _paths(self, dish_id: str) -> Tuple[Path, Path, Path]:
        split = "train" if dish_id in self.labels else "test"
        # load from three separate folders: color, depth_color, and depth_raw.
        rgb_path = self.data_root / split / "color" / dish_id / RGB_FILENAME
        depth_color_path = self.data_root / split / "depth_color" / dish_id / DEPTH_FILENAMES["depth_color"]
        depth_raw_path = self.data_root / split / "depth_raw" / dish_id / DEPTH_FILENAMES["depth_raw"]
        return rgb_path, depth_color_path, depth_raw_path

    def _sample_params(self, height: int, width: int) -> Tuple[float, Tuple[float, float], float, float, bool]:
        # These are the data augmentations (Rotation, Translation, etc.) listed in Table I.
        angle = random.uniform(-7.0, 7.0) if self.train else 0.0
        translate = (
            random.uniform(-0.05, 0.05) * width if self.train else 0.0,
            random.uniform(-0.05, 0.05) * height if self.train else 0.0,
        )
        scale = random.uniform(0.95, 1.05) if self.train else 1.0
        shear = random.uniform(-2.0, 2.0) if self.train else 0.0
        flip = random.random() < 0.5 if self.train else False
        return angle, translate, scale, shear, flip

    def _apply_transform(
        self,
        tensor: torch.Tensor,
        angle: float,
        translate: Tuple[float, float],
        scale: float,
        shear: float,
        flip: bool,
        interpolation: TF.InterpolationMode,
        *,
        pre_rotate: bool = False,
    ) -> torch.Tensor:
        if pre_rotate:
            # This implements preprocessing step to unify orientation (e.g., to portrait).
            tensor = TF.rotate(
                tensor, 90,
                interpolation=interpolation,
                expand=True,
                fill=0.0,
            )

        antialias = interpolation == TF.InterpolationMode.BILINEAR
        # apply the geometric augmentations from Table I.
        tensor = TF.affine(
            tensor,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[shear, 0.0],
            interpolation=interpolation,
            fill=0.0,
        )
        if flip:
            tensor = TF.hflip(tensor)

        size = self.input_size
        h, w = tensor.shape[-2:]
        scale_factor = size / max(h, w)
        nh, nw = max(1, int(round(h * scale_factor))), max(1, int(round(w * scale_factor)))
        tensor = TF.resize(tensor, [nh, nw], interpolation=interpolation, antialias=antialias)

        # pad the image to a 224x224 square, as described in setup.
        pad_h, pad_w = size - nh, size - nw
        pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2; pad_right = pad_w - pad_left
        tensor = TF.pad(tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=0.0)
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dish_id = self.dish_ids[index]
        rgb_path, depth_color_path, depth_raw_path = self._paths(dish_id)

        # load all the required modalities.
        rgb_np = load_rgb_image(rgb_path)
        depth_color_np, depth_color_mask = load_depth_color(depth_color_path)
        depth_raw_np, depth_raw_mask, _ = load_depth_raw(depth_raw_path)
        valid_mask = np.logical_or(depth_color_mask, depth_raw_mask)

        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)
        depth_color = torch.from_numpy(depth_color_np)
        depth_raw = torch.from_numpy(depth_raw_np)
        mask_tensor = torch.from_numpy(valid_mask.astype(np.float32))[None, ...]

        angle, translate, scale, shear, flip = self._sample_params(rgb.shape[1], rgb.shape[2])
        H, W = rgb.shape[1], rgb.shape[2]
        need_rot = False
        # This check enforces the canonical_orientation (e.g., portrait) from preprocessing step.
        if self.canonical_orientation == "portrait" and W > H:
            need_rot = True
        elif self.canonical_orientation == "landscape" and H > W:
            need_rot = True
        rgb = self._apply_transform(
            rgb, angle, translate, scale, shear, flip,
            TF.InterpolationMode.BILINEAR, pre_rotate=need_rot
        )
        # stack the depth modalities (color, raw) together.
        depth_stack = torch.cat([depth_color, depth_raw], dim=0)
        depth_stack = self._apply_transform(
            depth_stack, angle, translate, scale, shear, flip,
            TF.InterpolationMode.BILINEAR, pre_rotate=need_rot
        )
        mask_tensor = self._apply_transform(
            mask_tensor, angle, translate, scale, shear, flip,
            TF.InterpolationMode.NEAREST, pre_rotate=need_rot
        )
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
        # The final depth input is the stack of (depth_color, depth_raw, mask).
        # the widened depth input mention in report.
        depth_tensor = torch.cat([depth_stack, mask_tensor], dim=0)
        # normalize the RGB image using ImageNet statistics.
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        sample: Dict[str, torch.Tensor] = {
            "rgb": rgb,
            "depth": depth_tensor,
            "dish_id": dish_id,
        }

        if dish_id not in self.labels:
            return sample

        # If have labels, generate the ground truth targets for each head.
        total_calories = float(self.labels[dish_id])
        calorie_tensor = torch.tensor([total_calories], dtype=torch.float32)
        # Target for the main regression head (normalized).
        calorie_norm = self.target_normalizer.normalize(calorie_tensor.clone())
        bucket_idx = np.searchsorted(self.calorie_bins, total_calories, side="right") - 1
        bucket_idx = int(np.clip(bucket_idx, 0, len(self.calorie_bins) - 2))
        # Target for the segmentation head.
        segmentation = (mask_tensor > self.depth_threshold).float()
        mask_sum = float(segmentation.sum().item())
        if mask_sum < 1.0:
            segmentation = torch.ones_like(segmentation)
            mask_sum = float(segmentation.sum().item())
        calorie_density = total_calories / (mask_sum + 1e-6)
        # Target for the calorie density map head.
        calorie_map = segmentation * calorie_density

        # fetch the precomputed, normalized volume prior for this dish.
        volume_norm = self.volume_lookup.get(dish_id, 0.0)
        volume_actual = volume_norm * (self.volume_stats["std"] + 1e-6) + self.volume_stats["mean"]
        # This volume_tensor is the target for volume proxy head.
        volume_tensor = torch.tensor([volume_actual], dtype=torch.float32)

        targets: Dict[str, torch.Tensor] = {
            "segmentation": segmentation,
            "calorie_map": calorie_map,
            "total_calories": calorie_tensor,
            "total_calories_norm": calorie_norm,
            "volume": volume_tensor,
            "volume_norm": torch.tensor([volume_norm], dtype=torch.float32),
            "bucket": torch.tensor(bucket_idx, dtype=torch.long),
        }
        if self.macros is not None:
            macros = self.macros.get(dish_id, (0.0, 0.0, 0.0))
            targets["macros"] = torch.tensor(macros, dtype=torch.float32)
        sample["targets"] = targets
        return sample


def load_macros(csv_path: Optional[Path]) -> Optional[Dict[str, Tuple[float, ...]]]:
    """
    Load the macros from a CSV file.

    Args:
        csv_path: Path to the CSV file containing the macros.

    Returns:
        A dictionary of dish IDs to tuples of macros ("protein", "fat", "carbs", "carbohydrate").
    """
    if csv_path is None or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    possible_cols = [col for col in df.columns if col.lower() in {"protein", "fat", "carbs", "carbohydrate"}]
    if len(possible_cols) < 3:
        return None
    protein_col = next(col for col in possible_cols if col.lower() == "protein")
    fat_col = next(col for col in possible_cols if col.lower() == "fat")
    carb_col = next(col for col in possible_cols if col.lower() in {"carbs", "carbohydrate"})
    macros: Dict[str, Tuple[float, ...]] = {}
    for _, row in df.iterrows():
        dish_id = row["ID"]
        macros[dish_id] = (float(row[protein_col]), float(row[fat_col]), float(row[carb_col]))
    return macros


def split_ids(all_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split the dish IDs into training and validation sets.

    Args:
        all_ids: List of all dish IDs.
        val_ratio: Ratio of validation set size to the total size.
        seed: Random seed for random split.

    Returns:
        A tuple of (train_ids, val_ids).
    """   
    ids = list(all_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_size]
    train_ids = ids[val_size:]
    if not train_ids:
        raise ValueError("Training split is empty; adjust val_ratio.")
    return train_ids, val_ids


def prepare_dataloaders(
    config: TrainConfig,
) -> Tuple[
    DataLoader,
    DataLoader,
    TargetNormalizer,
    np.ndarray,
    Dict[str, float],
    Dict[str, float],
    torch.Tensor,
]:
    """
    Create train/val dataloaders and compute target mean.

    Args:
        config: Training config.

    Returns:
        A tuple of some initializable objects for training and evaluation, 
        including train_loader, val_loader, target_normalizer, calorie_bins, volume_lookup, volume_stats, bucket_weights.
    """ 
    if not config.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {config.train_csv}")
    labels_df = pd.read_csv(config.train_csv)
    label_map = dict(zip(labels_df["ID"], labels_df["Value"]))
    all_ids = list(label_map.keys())
    # use a 90%/10% random split for train/validation.
    train_ids, val_ids = split_ids(all_ids, config.val_ratio, config.seed)
    macros = load_macros(config.macro_csv) if config.predict_macros else None

    # where run the pre-computation for geometric volume prior.
    _, _, volume_lookup, volume_stats = compute_statistics(
        all_ids, label_map, config.data_root, config.num_bins
    )

    # compute the normalization stats and calorie bins only on the training set.
    target_normalizer, calorie_bins = compute_calorie_stats_for_ids(
        train_ids, label_map, config.num_bins
    )

    # also compute the weights for the optional calorie bucket head.
    bucket_weights = compute_bucket_weights(train_ids, label_map, calorie_bins)

    train_dataset = Nutrition5KMultitaskDataset(
        train_ids,
        data_root=config.data_root,
        input_size=config.input_size,
        train=True,
        target_normalizer=target_normalizer,
        calorie_bins=calorie_bins,
        volume_lookup=volume_lookup,
        volume_stats=volume_stats,
        labels=label_map,
        macros=macros,
        depth_threshold=config.depth_threshold,
        canonical_orientation=config.canonical_orientation,
    )
    val_dataset = Nutrition5KMultitaskDataset(
        val_ids,
        data_root=config.data_root,
        input_size=config.input_size,
        train=False,
        target_normalizer=target_normalizer,
        calorie_bins=calorie_bins,
        volume_lookup=volume_lookup,
        volume_stats=volume_stats,
        labels=label_map,
        macros=macros,
        depth_threshold=config.depth_threshold,
        canonical_orientation=config.canonical_orientation,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return (
        train_loader,
        val_loader,
        target_normalizer,
        calorie_bins,
        volume_lookup,
        volume_stats,
        bucket_weights,
    )


def build_model(config: TrainConfig, bucket_count: Optional[int] = None) -> RGBDVersion1:
    """
    Build the RGB-D model with encoders, fusion, and decoder heads.

    Args:
        config: Training configuration.
        bucket_count: Number of calorie classes, or None to disable.

    Returns:
        A ready-to-train RGBDVersion1 model.
    """
    # instantiates RGBDVersion1 (RGB-D+BiFPN+U-Net) model.
    return RGBDVersion1(
        rgb_backbone=config.rgb_backbone,
        depth_backbone=config.depth_backbone,
        depth_in_channels=config.depth_in_channels,
        fusion_channels=config.fusion_channels,
        num_bifpn_layers=config.bifpn_layers,
        decoder_channels=config.decoder_channels,
        predict_macros=config.predict_macros,
        bucket_count=bucket_count,
    )


def dict_to_device(targets: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    """
    Move a dict of tensors to a device. Keeps None as None.

    Args:
        targets: Dict of tensors or None.
        device: Destination device.
    """
    if targets is None:
        return None
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def compute_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_normalizer: TargetNormalizer,
) -> Dict[str, float]:
    """
    Compute simple metrics (MAE, RMSE, dice, etc.) for a batch.

    Args:
        outputs: Model outputs.
        targets: Ground truth tensors.
        target_normalizer: To denormalize calories.
    """
    metrics: Dict[str, float] = {}
    if "total_calories" in outputs and "total_calories" in targets:
        pred_norm = outputs["total_calories"].detach()
        # denormalize the predictions to compute metrics in the original kcal space.
        pred = target_normalizer.denormalize(pred_norm).cpu().view(-1)
        gt = targets["total_calories"].detach().cpu().view(-1)
        diff = torch.abs(pred - gt)
        # track MAE, which is one of primary evaluation metrics.
        metrics["calorie_mae"] = float(diff.mean())
        mse = torch.mean((pred - gt) ** 2)
        metrics["calorie_mse"] = float(mse)
        metrics["calorie_rmse"] = float(torch.sqrt(mse + 1e-8))
    if "volume" in outputs and "volume" in targets:
        # also track the error for volume proxy head.
        pred = outputs["volume"].detach().cpu().view(-1)
        gt = targets["volume"].detach().cpu().view(-1)
        diff = torch.abs(pred - gt)
        metrics["volume_mae"] = float(diff.mean())
    if "segmentation" in outputs and "segmentation" in targets:
        pred_mask = (torch.sigmoid(outputs["segmentation"]) > 0.5).float()
        gt_mask = targets["segmentation"]
        intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        union = (pred_mask + gt_mask).sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        metrics["segmentation_dice"] = float(dice.mean().detach().cpu())
    if "calorie_map" in outputs and "calorie_map" in targets:
        pred = outputs["calorie_map"].detach().cpu()
        gt = targets["calorie_map"].detach().cpu()
        l1 = torch.abs(pred - gt).view(pred.size(0), -1).mean(dim=1)
        metrics["calorie_map_l1"] = float(l1.mean())
    if "macros" in outputs and "macros" in targets:
        pred = outputs["macros"].detach().cpu()
        gt = targets["macros"].detach().cpu()
        l1 = torch.abs(pred - gt).mean(dim=1)
        metrics["macros_l1"] = float(l1.mean())
    if "calorie_logits" in outputs and "bucket" in targets:
        logits = outputs["calorie_logits"].detach().cpu()
        pred_cls = torch.argmax(logits, dim=-1)
        acc = (pred_cls == targets["bucket"].cpu().view(-1)).float().mean()
        metrics["bucket_acc"] = float(acc)
    return metrics


def merge_metric_logs(running: Dict[str, List[float]], batch_metrics: Dict[str, float]) -> None:
    """
    Append batch metric values into a list-accumulator dict.
    """
    for key, value in batch_metrics.items():
        running.setdefault(key, []).append(value)


def average_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute the mean of each metric list.

    Args:
        metrics: Dict of metric name to list of floats.
    """
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scaler: amp.GradScaler,
    criterion: MultitaskLoss,
    device: torch.device,
    max_grad_norm: float,
    target_normalizer: TargetNormalizer,
    log_interval: int,
) -> Dict[str, float]:
    """
    Train for one epoch and return averaged metrics.

    Runs forward, loss, backward, step with AMP and grad clip.
    """
    model.train()
    running_loss = 0.0
    running_metrics: Dict[str, List[float]] = {}
    total_samples = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="train", leave=False), start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        targets = dict_to_device(batch["targets"], device)
        if targets is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            # run the forward pass through full RGBD+BiFPN+U-Net model.
            outputs = model(rgb, depth)
            # compute the combined multi-task loss.
            loss, loss_breakdown = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = rgb.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        merge_metric_logs(running_metrics, loss_breakdown)
        # compute and log metrics (like MAE) for this batch.
        batch_metrics = compute_batch_metrics(outputs, targets, target_normalizer)
        merge_metric_logs(running_metrics, batch_metrics)
        if batch_idx % max(1, log_interval) == 0:
            LOGGER.info(
                "train step %d/%d | loss %.4f | calorie_mse %.4f | calorie_mae %.4f",
                batch_idx,
                len(dataloader),
                loss.item(),
                batch_metrics.get("calorie_mse", float("nan")),
                batch_metrics.get("calorie_mae", float("nan")),
            )
    if total_samples == 0:
        return {"loss": float("nan")}
    epoch_loss = running_loss / total_samples
    metrics = average_metrics(running_metrics)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultitaskLoss,
    device: torch.device,
    target_normalizer: TargetNormalizer,
) -> Dict[str, float]:
    """
    Run evaluation on a dataloader and return averaged metrics.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    running_metrics: Dict[str, List[float]] = {}
    true_hist = None
    pred_hist = None
    n_bins = None
    for batch in tqdm(dataloader, desc="val", leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        targets = dict_to_device(batch["targets"], device)
        if targets is None:
            continue
        outputs = model(rgb, depth)
        loss, loss_breakdown = criterion(outputs, targets)
        batch_size = rgb.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        merge_metric_logs(running_metrics, loss_breakdown)
        merge_metric_logs(running_metrics, compute_batch_metrics(outputs, targets, target_normalizer))
        if "calorie_logits" in outputs and "bucket" in targets:
            logits = outputs["calorie_logits"].detach().cpu()
            pred_cls = torch.argmax(logits, dim=-1)
            tb = targets["bucket"].detach().cpu().view(-1)
            n_bins = logits.shape[-1]
            if true_hist is None:
                true_hist = torch.zeros(n_bins, dtype=torch.long)
                pred_hist = torch.zeros(n_bins, dtype=torch.long)
            true_hist += torch.bincount(tb, minlength=n_bins)
            pred_hist += torch.bincount(pred_cls, minlength=n_bins)
    if total_samples == 0:
        return {"loss": float("nan")}
    metrics = average_metrics(running_metrics)
    metrics["loss"] = running_loss / total_samples
    if true_hist is not None:
        LOGGER.info(f"bucket true hist: {true_hist.tolist()}")
        LOGGER.info(f"bucket pred  hist: {pred_hist.tolist()}")
    return metrics

class ProjectionHead(nn.Module):
    # the 2-layer MLP projection head used in self-supervised pretraining stage.
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # It projects the global average pooled features before the contrastive loss.
        return self.net(x)


class ReconstructionHead(nn.Module):
    # an auxiliary head for the self-supervised stage, used to reconstruct masked depth.
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden1 = max(in_channels // 2, 32)
        hidden2 = max(hidden1 // 2, 16)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden1, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden2, 1, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.decoder(feature_map)


def info_nce(rgb_proj: torch.Tensor, depth_proj: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss between two embeddings.

    Args:
        rgb_proj: RGB embeddings (B,D).
        depth_proj: Depth embeddings (B,D).
        temperature: Softmax temperature.
    """
    # implements the InfoNCE objective (contrastive loss) from report.
    rgb_norm = F.normalize(rgb_proj, dim=1)
    depth_norm = F.normalize(depth_proj, dim=1)
    logits = rgb_norm @ depth_norm.t() / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    # enforce that the RGB and Depth projections of the same dish are similar (positive pairs).
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


class SelfSupervisedTrainer:
    # This class manages the self-supervised pretraining stage (Step 'a' in Fig. 1).
    def __init__(
        self,
        model: RGBDVersion1,
        config: TrainConfig,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.device = device
        rgb_dim = self.model.rgb_encoder.out_channels[-1]
        depth_dim = self.model.depth_encoder.out_channels[-1]
        # attach the projection heads to the encoders.
        self.rgb_proj = ProjectionHead(rgb_dim).to(device)
        self.depth_proj = ProjectionHead(depth_dim).to(device)
        self.reconstruction_head = ReconstructionHead(depth_dim).to(device)
        # only optimize the encoders and the new projection heads during this stage.
        params = (
            list(self.model.rgb_encoder.parameters())
            + list(self.model.depth_encoder.parameters())
            + list(self.rgb_proj.parameters())
            + list(self.depth_proj.parameters())
            + list(self.reconstruction_head.parameters())
        )
        lr = self.config.selfsup_learning_rate or self.config.learning_rate
        self.optimizer = AdamW(params, lr=lr, weight_decay=self.config.weight_decay)
        self.scaler = create_grad_scaler(self.config.use_amp)

    def train(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        if self.config.selfsup_epochs <= 0:
            return history
        self.model.train()
        self.rgb_proj.train()
        self.depth_proj.train()
        self.reconstruction_head.train()
        LOGGER.info("Starting self-supervised pretraining phase...")
        for epoch in range(self.config.selfsup_epochs):
            epoch_loss = 0.0
            contrast_loss = 0.0
            recon_loss = 0.0
            steps = 0
            progress = tqdm(
                self.dataloader,
                desc=f"self-sup {epoch + 1}/{self.config.selfsup_epochs}",
                leave=False,
            )
            for batch in progress:
                rgb = batch["rgb"].to(self.device, non_blocking=True)
                depth_full = batch["depth"].to(self.device, non_blocking=True)
                # separate the depth stack into features and masks.
                mask_idx = depth_full.shape[1] - 1
                raw_idx = depth_full.shape[1] - 2
                depth_features = depth_full[:, :mask_idx]
                depth_mask_channel = depth_full[:, mask_idx: mask_idx + 1]
                depth_raw = depth_full[:, raw_idx: raw_idx + 1]
                # apply the random patch-based depth masking.
                masked_depth, keep_mask = apply_depth_patch_mask(
                    depth_features, self.config.selfsup_depth_mask_ratio, self.config.selfsup_patch_size
                )
                depth_input = torch.cat([masked_depth, depth_mask_channel], dim=1)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    # get features from both encoders.
                    rgb_feats = self.model.rgb_encoder(rgb)
                    depth_feats = self.model.depth_encoder(depth_input)
                    # pool and project them for the contrastive loss.
                    rgb_global = F.adaptive_avg_pool2d(rgb_feats[-1], 1).flatten(1)
                    depth_global = F.adaptive_avg_pool2d(depth_feats[-1], 1).flatten(1)
                    rgb_proj = self.rgb_proj(rgb_global)
                    depth_proj = self.depth_proj(depth_global)
                    # InfoNCE contrastive loss.
                    loss_contrast = info_nce(rgb_proj, depth_proj, self.config.selfsup_temperature)
                    # auxiliary depth reconstruction loss.
                    depth_map = depth_feats[-1]
                    recon = self.reconstruction_head(depth_map)
                    target = F.interpolate(depth_raw, size=recon.shape[-2:], mode="bilinear", align_corners=False)
                    valid_mask = F.interpolate(depth_mask_channel, size=recon.shape[-2:], mode="nearest")
                    keep_mask_low = F.interpolate(keep_mask, size=recon.shape[-2:], mode="nearest")
                    # only compute the reconstruction loss on the patches that were masked out.
                    missing = (1.0 - keep_mask_low) * valid_mask
                    weight = missing if missing.sum() > 1e-6 else valid_mask
                    recon_diff = torch.abs(recon - target) * weight
                    loss_recon = recon_diff.sum() / (weight.sum() + 1e-6)
                    # The total self-supervised loss is a weighted sum of both.
                    loss = loss_contrast + self.config.selfsup_reconstruction_weight * loss_recon
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                steps += 1
                epoch_loss += loss.item()
                contrast_loss += loss_contrast.item()
                recon_loss += loss_recon.item()
                progress.set_postfix(
                    total=f"{loss.item():.4f}", contrast=f"{loss_contrast.item():.4f}", recon=f"{loss_recon.item():.4f}"
                )
            if steps == 0:
                continue
            avg_total = epoch_loss / steps
            avg_contrast = contrast_loss / steps
            avg_recon = recon_loss / steps
            LOGGER.info(
                "Self-supervised epoch %d/%d | total %.4f | contrast %.4f | recon %.4f",
                epoch + 1, self.config.selfsup_epochs, avg_total, avg_contrast, avg_recon
            )
            history.append({
                "epoch": epoch + 1,
                "loss_total": avg_total,
                "loss_contrast": avg_contrast,
                "loss_reconstruction": avg_recon,
            })
        # After this stage, the pretrained encoder weights are transferred to the main model.
        return history

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scaler: amp.GradScaler,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model, optimizer, scaler, and extra state to a file.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scaler: AMP scaler state.
        epoch: Current epoch number.
        metrics: Metrics to store.
        path: Output file path.
        extra_state: Any extra info to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra_state:
        # store metadata like the normalizer and volume stats in the checkpoint.
        state["extra_state"] = extra_state
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    scaler: Optional[amp.GradScaler],
    path: Path,
) -> Tuple[int, Dict[str, Any]]:
    """
    Load model (and optional optimizer/scaler) from a checkpoint file.

    Args:
        model: Model to load into.
        optimizer: Optional optimizer to load.
        scaler: Optional AMP scaler to load.
        path: Checkpoint path.

    Returns:
        Tuple of (start_epoch, extra_state dict).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    extra = checkpoint.get("extra_state", {})
    return int(checkpoint.get("epoch", 0)), extra


def parse_args() -> TrainConfig:
    """
    Parse CLI args and build a TrainConfig.

    Returns:
        A populated TrainConfig object.
    """
    parser = argparse.ArgumentParser(description=" RGB-D Version1 multitask trainer")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("comp-90086-nutrition-5-k") / "Nutrition5K" / "Nutrition5K",
        help="Root directory of Nutrition5K dataset.",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="depth_color",
        choices=list(DEPTH_FILENAMES.keys()),
        help="Depth modality to use.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV with training calories. Defaults to nutrition5k_train.csv under data-root.",
    )
    parser.add_argument(
        "--macro-csv",
        type=Path,
        default=None,
        help="Optional CSV containing macro nutrient labels (ID, protein, fat, carbs).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--bifpn-layers", type=int, default=2)
    parser.add_argument(
        "--decoder-channels",
        type=int,
        nargs="+",
        default=[160, 112, 80],
        help="Decoder channel widths from coarse to fine.",
    )
    parser.add_argument("--rgb-backbone", type=str, default="resnet34")
    parser.add_argument("--depth-backbone", type=str, default="resnet34")
    parser.add_argument("--depth-in-channels", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--depth-threshold", type=float, default=0.9)
    parser.add_argument("--volume-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "rgbd_bifpn_unet_selfsup")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--no-macros",
        action="store_true",
        help="Disable macro nutrient head.",
    )
    parser.add_argument("--num-bins", type=int, default=2, help="Number of calorie quantisation bins.")
    parser.add_argument(
        "--fuse-alpha",
        type=float,
        default=1.0,
        help="Fusion weight between regression and classification expectation.",
    )
    parser.add_argument(
        "--canonical-orientation",
        type=str,
        default="portrait",
        choices=["none", "portrait", "landscape"],
        help="Canonical orientation for images before scaling.",
    )
    parser.add_argument("--log-interval", type=int, default=30, help="Steps between train log messages.")
    parser.add_argument("--self-sup-epochs", type=int, default=20, help="Number of epochs for self-supervised pretraining.")
    parser.add_argument("--self-sup-temperature", type=float, default=0.2, help="Contrastive temperature for RGB-depth alignment.")
    parser.add_argument("--self-sup-reconstruction-weight", type=float, default=0.5, help="Weight for depth reconstruction loss during self-supervised training.")
    parser.add_argument("--self-sup-depth-mask-ratio", type=float, default=0.5, help="Fraction of depth patches to drop when creating masked depth inputs.")
    parser.add_argument("--self-sup-patch-size", type=int, default=8, help="Patch size (in pixels) used for depth masking.")
    parser.add_argument("--self-sup-lr", type=float, default=None, help="Learning rate for self-supervised pretraining (defaults to --lr).")
    parser.add_argument(
        "--history-filename",
        type=str,
        default="history.jsonl",
        help="Filename (under output-dir) for per-epoch metrics JSONL.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=30,
        help="Save checkpoint and run evaluation/prediction every N epochs.",
    )
    args = parser.parse_args()
    train_csv = args.train_csv or (args.data_root / "nutrition5k_train.csv")
    config = TrainConfig(
        data_root=args.data_root,
        depth_source=args.depth,
        train_csv=train_csv,
        macro_csv=args.macro_csv,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        rgb_backbone=args.rgb_backbone,
        depth_backbone=args.depth_backbone,
        fusion_channels=args.fusion_channels,
        bifpn_layers=args.bifpn_layers,
        decoder_channels=tuple(args.decoder_channels),
        predict_macros=not args.no_macros,
        depth_in_channels=args.depth_in_channels,
        input_size=args.input_size,
        max_grad_norm=args.max_grad_norm,
        depth_threshold=args.depth_threshold,
        volume_scale=args.volume_scale,
        num_bins=args.num_bins,
        log_interval=args.log_interval,
        fuse_alpha=args.fuse_alpha,
        canonical_orientation=args.canonical_orientation,
        selfsup_epochs=args.self_sup_epochs,
        selfsup_temperature=args.self_sup_temperature,
        selfsup_reconstruction_weight=args.self_sup_reconstruction_weight,
        selfsup_depth_mask_ratio=args.self_sup_depth_mask_ratio,
        selfsup_patch_size=args.self_sup_patch_size,
        selfsup_learning_rate=args.self_sup_lr,
        history_filename=args.history_filename,
        save_interval=args.save_interval,
    )
    return config


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format a metrics dict into a short string.

    Args:
        metrics: Dict of metric name to float.
    """
    return ", ".join(f"{key}: {value:.4f}" for key, value in sorted(metrics.items()))


def train_and_evaluate(config: TrainConfig) -> None:
    """
    Main training loop with optional self-supervised warmup.

    Trains, evaluates, logs, and saves checkpoints.
    """
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logging(config.output_dir)
    (
        train_loader,
        val_loader,
        target_normalizer,
        calorie_bins,
        volume_lookup,
        volume_stats,
        bucket_weights,
    ) = prepare_dataloaders(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    def refresh_dataset_state() -> None:
        datasets = [
            getattr(train_loader, "dataset", None),
            getattr(val_loader, "dataset", None),
        ]
        for dataset in datasets:
            if isinstance(dataset, Nutrition5KMultitaskDataset):
                dataset.target_normalizer = target_normalizer
                dataset.calorie_bins = calorie_bins
                dataset.volume_lookup = volume_lookup
                dataset.volume_stats = volume_stats

    refresh_dataset_state()

    bin_midpoints_norm = make_bin_midpoints_norm(calorie_bins, target_normalizer)

    effective_bucket_count = max(0, int(len(calorie_bins) - 1))
    if effective_bucket_count <= 1:
        LOGGER.warning(
            "Effective bucket count = %d (<=1). Calorie bucket head will be disabled.",
            effective_bucket_count
        )
    else:
        LOGGER.info("Using equal-frequency bins: %d buckets", effective_bucket_count)
        LOGGER.info("Calorie bin edges (first 10 shown): %s",
                    np.array2string(calorie_bins[:10], precision=2, floatmode='fixed'))
    model = build_model(config, bucket_count=effective_bucket_count).to(device)

    # run self-supervised pretraining only if specified and if are not loading a checkpoint.
    run_selfsup = (
        config.selfsup_epochs > 0 and not (config.checkpoint_path is not None and config.checkpoint_path.exists())
    )
    if run_selfsup:
        LOGGER.info("Starting self-supervised pretraining for %d epochs", config.selfsup_epochs)
        selfsup_trainer = SelfSupervisedTrainer(model, config, train_loader, device)
        pretrain_history = selfsup_trainer.train()
        for entry in pretrain_history:
            metrics = {key: value for key, value in entry.items() if key != "epoch"}
            append_history_entry(
                config.output_dir,
                config.history_filename,
                {"epoch": entry["epoch"], "phase": "selfsup", "metrics": metrics},
            )
        LOGGER.info("Self-supervised pretraining complete. Transferring encoder weights.")
    elif config.selfsup_epochs > 0:
        LOGGER.info(
            "Skipping self-supervised pretraining because checkpoint %s will be loaded.",
            config.checkpoint_path,
        )
    # initialize multi-task loss function.
    criterion = MultitaskLoss(
        calorie_bins=calorie_bins,
        target_normalizer=target_normalizer,
        bucket_weights=bucket_weights,
        ce_temperature=1.0,
        ce_label_smoothing=0.05,
    )
    # set initial weights for the auxiliary losses.
    criterion.loss_weights.update({
        "calorie_bucket": 0.30,
        "calorie_expectation": 0.35,
        "calorie_from_map": 0.35,
    })
    _base_w_bucket = criterion.loss_weights.get("calorie_bucket", 0.30)
    _base_w_expect = criterion.loss_weights.get("calorie_expectation", 0.25)
    _base_w_frommp = criterion.loss_weights.get("calorie_from_map", 0.30)
    criterion = criterion.to(device)
    # use the AdamW optimizer as mentioned in report.
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = create_grad_scaler(config.use_amp)
    # set the epoch at which switch from SmoothL1 to MSE loss.
    switch_to_mse_epoch = 50

    # use the ReduceLROnPlateau scheduler, as described in experimental setup.
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.patience,
        cooldown=2, min_lr=1e-6, threshold=1e-3
    )
    start_epoch = 0
    best_metric = float("inf")
    best_path = config.output_dir / "best_model.pt"

    def build_extra_state() -> Dict[str, Any]:
        # save computed statistics (normalizer, bins, volume) with the model.
        state = {
            "config": asdict(config),
            "target_normalizer": asdict(target_normalizer),
            "calorie_bins": calorie_bins.tolist(),
            "volume_stats": volume_stats,
            "volume_lookup": volume_lookup,
        }
        return state

    if config.checkpoint_path is not None and config.checkpoint_path.exists():
        start_epoch, extra_state = load_checkpoint(model, optimizer, scaler, config.checkpoint_path)
        LOGGER.info("Loaded checkpoint from %s (epoch %d)", config.checkpoint_path, start_epoch)
        # restore the statistics from the checkpoint.
        if extra_state:
            tn_state = extra_state.get("target_normalizer")
            if tn_state:
                target_normalizer = TargetNormalizer(**tn_state)
            if "calorie_bins" in extra_state:
                calorie_bins = np.array(extra_state["calorie_bins"], dtype=np.float32)
            if "volume_stats" in extra_state:
                volume_stats = extra_state["volume_stats"]
            if "volume_lookup" in extra_state:
                volume_lookup.update(extra_state["volume_lookup"])
            refresh_dataset_state()
            try:
                effective_bucket_count = max(0, int(len(calorie_bins) - 1))
                criterion.set_calorie_bins(calorie_bins, target_normalizer)
                criterion.set_target_normalizer(target_normalizer)
                LOGGER.info("Restored equal-frequency bins: %d buckets", effective_bucket_count)
            except Exception:
                pass
            bin_midpoints_norm = make_bin_midpoints_norm(calorie_bins, target_normalizer)

    # the main training loop.
    for epoch in range(start_epoch, config.epochs):
        # use a linear warm-up for the auxiliary task weights, as described in report.
        warm = min(1.0, (epoch + 1) / 5.0)
        criterion.loss_weights["calorie_bucket"] = _base_w_bucket * warm
        criterion.loss_weights["calorie_expectation"] = _base_w_expect * warm
        criterion.loss_weights["calorie_from_map"] = _base_w_frommp * warm
        LOGGER.info("Epoch %d/%d", epoch + 1, config.epochs)
        t0 = time.time()
        t_train0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            config.max_grad_norm,
            target_normalizer,
            config.log_interval,
        )
        train_metrics["time_sec"] = float(time.time() - t_train0)
        try:
            train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        LOGGER.info("Train: %s", format_metrics(train_metrics))
        append_history_entry(
            config.output_dir,
            config.history_filename,
            {"epoch": epoch + 1, "phase": "train", "metrics": train_metrics},
        )
        t_val0 = time.time()
        val_metrics = evaluate(model, val_loader, criterion, device, target_normalizer)
        val_metrics["time_sec"] = float(time.time() - t_val0)
        try:
            val_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        # use validation RMSE (or MAE/loss as fallback) to guide the scheduler and for early stopping.
        scheduler_metric = val_metrics.get(
            "calorie_rmse", val_metrics.get("calorie_mae", val_metrics.get("loss", float("inf")))
        )
        try:
            val_metrics["scheduler_metric"] = float(scheduler_metric)
        except Exception:
            pass
        LOGGER.info("Val  : %s", format_metrics(val_metrics))
        append_history_entry(
            config.output_dir,
            config.history_filename,
            {"epoch": epoch + 1, "phase": "val", "metrics": val_metrics},
        )
        scheduler.step(scheduler_metric)
        extra_state = build_extra_state()
        save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch + 1,
            val_metrics,
            config.output_dir / "last_model.pt",
            extra_state=extra_state,
        )

        # This implements Loss Design strategy from report.
        if (epoch + 1) == switch_to_mse_epoch:
            # switch the main loss from SmoothL1 to MSE.
            criterion.set_calorie_total_loss(kind="mse", use_kcal_space=True)
            # also update the loss weights to focus more on the main MSE loss.
            criterion.loss_weights.update({
                "calorie_total": 1.0,
                "calorie_bucket": 0.10,
                "calorie_expectation": 0.10,
                "calorie_from_map": 0.10,
                "segmentation": 0.02,
                "macros": 0.0,
                "volume": 0.0,
            })
            _base_w_bucket = criterion.loss_weights["calorie_bucket"]
            _base_w_expect = criterion.loss_weights["calorie_expectation"]
            _base_w_frommp = criterion.loss_weights["calorie_from_map"]
            LOGGER.info("Switched calorie_total loss to MSE in kcal space at epoch %d; "
                        "rebalanced aux head weights for MSE-focused fine-tuning.", epoch + 1)

        # save the checkpoint if it has the best validation score so far.
        if scheduler_metric < best_metric:
            best_metric = scheduler_metric
            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch + 1,
                val_metrics,
                best_path,
                extra_state=extra_state,
            )
            LOGGER.info("New best model with score %.4f", best_metric)
        if (epoch + 1) % max(1, config.save_interval) == 0:
            tag = f"epoch_{epoch + 1}"
            periodic_path = config.output_dir / f"checkpoint_{tag}.pt"
            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch + 1,
                val_metrics,
                periodic_path,
                extra_state=extra_state,
            )
            t0 = time.time()
            val_metrics = evaluate(model, val_loader, criterion, device, target_normalizer)
            val_metrics["time_sec"] = float(time.time() - t0)
            val_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            append_history_entry(
                config.output_dir,
                config.history_filename,
                {"epoch": epoch, "phase": f"val_periodic_{tag}", "metrics": val_metrics},
            )

    LOGGER.info("Training complete.")
    if best_path.exists():
        # load the best performing checkpoint for final evaluation.
        _, extra_state = load_checkpoint(model, None, None, best_path)
        LOGGER.info("Loaded best model from %s", best_path)
        if extra_state:
            tn_state = extra_state.get("target_normalizer")
            if tn_state:
                target_normalizer = TargetNormalizer(**tn_state)
            if "calorie_bins" in extra_state:
                calorie_bins = np.array(extra_state["calorie_bins"], dtype=np.float32)
            if "volume_stats" in extra_state:
                volume_stats = extra_state["volume_stats"]
            if "volume_lookup" in extra_state:
                volume_lookup.update(extra_state["volume_lookup"])
            refresh_dataset_state()
            try:
                criterion.set_calorie_bins(calorie_bins, target_normalizer)
                criterion.set_target_normalizer(target_normalizer)
            except Exception:
                pass
            bin_midpoints_norm = make_bin_midpoints_norm(calorie_bins, target_normalizer)
    final_metrics = evaluate(model, val_loader, criterion, device, target_normalizer)
    LOGGER.info("Best validation metrics: %s", format_metrics(final_metrics))

def main() -> None:
    config = parse_args()
    train_and_evaluate(config)


if __name__ == "__main__":
    main()