import argparse
import json
import logging
import math
import random
import sys
import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.cuda import amp
from torch.optim import RMSprop, AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_root: Path
    modality: str
    model_name: str
    epochs: int
    batch_size: int
    val_batch_size: int
    val_ratio: float
    seed: int
    num_workers: int
    use_amp: bool
    output_dir: Path
    checkpoint_path: Optional[Path]
    depth_source: str
    learning_rate: float
    weight_decay: float
    patience: int
    log_interval: int
    history_filename: str
    save_interval: int
    # These parameters control the self-supervised pretraining for baseline model.
    selfsup_epochs: int
    selfsup_temperature: float
    selfsup_learning_rate: Optional[float]
    # This corresponds to the random mask occlusion to the depth channel for RGBD+SelfSup baseline.
    selfsup_depth_mask_ratio: float
    selfsup_patch_size: int


RGB_FILENAME = "rgb.png"
DEPTH_FILENAMES = {
    "depth_color": "depth_color.png",
    "depth_raw": "depth_raw.png",
}

# use standard ImageNet normalization for the RGB channels.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def append_history_entry(output_dir: Path, filename: str, entry: Dict[str, object]) -> None:
    """
    Append a single history record to JSONL and a wide CSV for plotting.

    JSONL keeps the raw entry, while CSV flattens metrics so each row has
    columns: epoch, phase, and metric keys as columns.
    """
    # JSONL
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    # CSV (wide format)
    try:
        csv_name = Path(filename).with_suffix(".csv").name
        csv_path = output_dir / csv_name
        base = {"epoch": entry.get("epoch"), "phase": entry.get("phase")}
        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        metrics = metrics or {}
        row: Dict[str, float] = {**base}  # type: ignore[assignment]
        for k, v in metrics.items():  # type: ignore[assignment]
            if isinstance(v, (int, float, np.floating)) and not (
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            ):
                row[k] = float(v)
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            try:
                df_old = pd.read_csv(csv_path)
                for col in df_new.columns:
                    if col not in df_old.columns:
                        df_old[col] = np.nan
                for col in df_old.columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan
                df_all = pd.concat([df_old, df_new[df_old.columns]], ignore_index=True)
            except Exception:
                df_all = df_new
        else:
            df_all = df_new
        cols = [c for c in df_all.columns if c not in ("epoch", "phase")]
        ordered_cols = ["epoch", "phase", *sorted(cols)]
        df_all.to_csv(csv_path, index=False, columns=ordered_cols)
    except Exception as e:
        LOGGER.warning("Failed to update history CSV: %s", e)


class Nutrition5KRGBDDataset(Dataset):
    # This dataset class implements the single tower fusion (early fusion) baseline model.
    def __init__(
        self,
        dish_ids: Sequence[str],
        data_root: Path,
        depth_source: str,
        input_size: int,
        labels: Optional[Dict[str, float]] = None,
        train: bool = True,
    ) -> None:
        self.dish_ids = list(dish_ids)
        self.data_root = data_root
        self.depth_source = depth_source
        self.input_size = input_size
        self.labels = labels
        self.train = train

        self.rgb_mean = IMAGENET_MEAN
        self.rgb_std = IMAGENET_STD
        # use simple normalization for the single-channel depth map in this baseline.
        self.depth_mean = (0.5,)
        self.depth_std = (0.5,)

    def __len__(self) -> int:
        return len(self.dish_ids)

    def _paths(self, dish_id: str) -> Tuple[Path, Path]:
        split = "train" if self.labels is not None else "test"
        rgb_path = self.data_root / split / "color" / dish_id / RGB_FILENAME
        depth_path = self.data_root / split / self.depth_source / dish_id / DEPTH_FILENAMES[self.depth_source]
        return rgb_path, depth_path

    def _get_crop_params(self, w: int, h: int) -> Tuple[int, int, int, int, bool]:
        scale = (0.8, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        i, j, th, tw = transforms.RandomResizedCrop.get_params(Image.new("RGB", (w, h)), scale, ratio)
        do_flip = random.random() < 0.5
        return i, j, th, tw, do_flip

    def _apply_spatial(self, img: Image.Image, i: int, j: int, th: int, tw: int, do_flip: bool) -> Image.Image:
        img = img.crop((j, i, j + tw, i + th))
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        if do_flip:
            # apply a horizontal flip augmentation.
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _normalize(self, t: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c].sub_(m).div_(s)
        return t

    def __getitem__(self, index: int):
        dish_id = self.dish_ids[index]
        rgb_path, depth_path = self._paths(dish_id)
        with Image.open(rgb_path) as im_rgb, Image.open(depth_path) as im_d:
            im_rgb = im_rgb.convert("RGB")
            # convert depth to grayscale ('L') as it's a single channel.
            im_d = im_d.convert("L")

            long = int(self.input_size * 1.1)
            im_rgb = transforms.Resize(long)(im_rgb)
            im_d = transforms.Resize(long, interpolation=Image.NEAREST)(im_d)

            if self.train:
                # apply consistent random cropping and flipping to both modalities.
                w, h = im_rgb.size
                i, j, th, tw, do_flip = self._get_crop_params(w, h)
                im_rgb = self._apply_spatial(im_rgb, i, j, th, tw, do_flip)
                im_d = self._apply_spatial(im_d, i, j, th, tw, do_flip)
                im_rgb = transforms.ColorJitter(0.2, 0.2, 0.2, 0.02)(im_rgb)
            else:
                im_rgb = transforms.CenterCrop(self.input_size)(im_rgb)
                im_d = transforms.CenterCrop(self.input_size)(im_d)

            t_rgb = transforms.ToTensor()(im_rgb)
            # normalize the RGB channels using ImageNet stats.
            t_rgb = self._normalize(t_rgb, self.rgb_mean, self.rgb_std)
            t_d = transforms.ToTensor()(im_d)  # [1,H,W]
            t_d = self._normalize(t_d, self.depth_mean, self.depth_std)
            # the early fusion step, creating the 4-channel input for baseline model.
            x = torch.cat([t_rgb, t_d], dim=0)

        if self.labels is None:
            return x, dish_id
        return x, torch.tensor(self.labels[dish_id], dtype=torch.float32)

def adapt_backbone_for_4ch(model: nn.Module) -> nn.Module:
    """
    Replace first conv to accept 4 channels by copying RGB weights.

    Args:
        model: Backbone with conv1 field.

    Returns:
        The modified model.
    """
    # adapt the ResNet-34 backbone to accept 4-channel (RGB+D) input.
    old = model.conv1
    new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        # copy the original RGB weights for the first 3 channels.
        new.weight[:, :3] = old.weight
        # initialize the new (4th) channel's weights by averaging the RGB weights.
        new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.conv1 = new
    return model


def forward_with_optional_aux(
    model: nn.Module,
    images: torch.Tensor,
    targets: Optional[torch.Tensor],
    criterion: Optional[nn.Module],
    include_aux_loss: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward pass handling optional aux logits and loss.

    Args:
        model: Model that may return main and aux outputs.
        images: Input batch.
        targets: Targets for loss, or None.
        criterion: Loss function, or None.
        include_aux_loss: If true, add aux loss with weight 0.4.

    Returns:
        Tuple of (main_output, aux_output, loss or None).
    """
    outputs = model(images)
    main_output = outputs
    aux_output = None
    if hasattr(outputs, "logits") and hasattr(outputs, "aux_logits"):
        main_output = outputs.logits
        aux_output = outputs.aux_logits
    elif isinstance(outputs, (tuple, list)):
        main_output = outputs[0]
        aux_output = outputs[1] if len(outputs) > 1 else None

    # squeeze the output to get a single calorie value per sample.
    main_output = main_output.squeeze(-1)
    if main_output.dim() > 1:
        main_output = main_output[:, 0]

    loss = None
    if targets is not None and criterion is not None:
        loss = criterion(main_output, targets)
        if include_aux_loss and aux_output is not None:
            aux_logits = aux_output.squeeze(-1)
            if aux_logits.dim() > 1:
                aux_logits = aux_logits[:, 0]
            loss = loss + 0.4 * criterion(aux_logits, targets)
    return main_output, aux_output, loss


def compute_metrics(
    predictions: Iterable[float],
    targets: Iterable[float],
    target_mean: float,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and relative error (percent).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        target_mean: Mean of targets for relative error.

    Returns:
        Dict with keys: mae, rmse, rel_err_pct.
    """
    preds = np.array(list(predictions), dtype=np.float32)
    gts = np.array(list(targets), dtype=np.float32)
    # use Mean Absolute Error (MAE) as one of primary evaluation metrics.
    mae = float(np.mean(np.abs(preds - gts)))
    mse = float(np.mean((preds - gts) ** 2))
    rel_err = float(mae / target_mean * 100.0) if target_mean > 0 else float("nan")
    return {
        "mae": mae,
        # also report RMSE, which is used in Table II.
        "rmse": float(np.sqrt(mse)),
        "rel_err_pct": rel_err,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format a metrics dict into a short string.

    Args:
        metrics: Dict of metric name to float.
    """
    return ", ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))


def apply_depth_patch_mask_on_4ch(
    x: torch.Tensor,
    ratio: float,
    patch_size: int,
) -> torch.Tensor:
    """
    Apply patch masking on the depth channel of a 4-ch RGB-D tensor.

    Args:
        x: Input tensor [B,4,H,W].
        ratio: Fraction of patches to drop.
        patch_size: Patch size in pixels.

    Returns:
        New tensor with masked depth channel.
    """
    # This implements the random mask occlusion to the depth channel for SelfSup baseline.
    if ratio <= 0.0 or x.size(1) < 4:
        return x
    b, c, h, w = x.shape
    p = max(1, patch_size)
    gh, gw = h // p, w // p
    if gh == 0 or gw == 0:
        return x
    total = gh * gw
    keep = max(1, int(round((1.0 - ratio) * total)))
    mask_flat = x.new_zeros((b, total))
    for i in range(b):
        idx = torch.randperm(total, device=x.device)[:keep]
        mask_flat[i, idx] = 1.0
    mask = mask_flat.view(b, 1, gh, gw)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
    out = x.clone()
    # apply the mask *only* to the 4th channel, which is depth data.
    out[:, 3:4] = out[:, 3:4] * mask
    return out


def random_tensor_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Lightweight tensor-level augmentation for 4-ch inputs.

    Applies random flip, light color jitter, and gaussian noise.

    Args:
        x: Input tensor [B,4,H,W].

    Returns:
        Augmented tensor.
    """
    # apply simple augmentations to create a new view of the input for contrastive learning.
    b, c, h, w = x.shape
    out = x.clone()
    # Flip
    if random.random() < 0.5:
        out = torch.flip(out, dims=[-1])
    # Color jitter (scale + bias)
    if c >= 3:
        scale = out.new_tensor([random.uniform(0.9, 1.1) for _ in range(3)]).view(1, 3, 1, 1)
        bias = out.new_tensor([random.uniform(-0.05, 0.05) for _ in range(3)]).view(1, 3, 1, 1)
        out[:, :3] = out[:, :3] * scale + bias
    # Noise
    noise = out.new_zeros(out.shape).normal_(mean=0.0, std=0.01)
    out = out + noise
    return out


class FeatureHook:
    # a helper class to extract intermediate features from the model.
    def __init__(self, module: nn.Module) -> None:
        self.tensor: Optional[torch.Tensor] = None
        self.h = module.register_forward_hook(self.hook)

    def hook(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self.tensor = out.detach()

    def close(self) -> None:
        self.h.remove()


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
        return self.net(x)


def info_nce_views(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss between two embeddings.

    Args:
        z1: First embeddings (B,D).
        z2: Second embeddings (B,D).
        temperature: Softmax temperature.
    """
    # implements the InfoNCE objective for contrastive pretraining.
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.t() / max(temperature, 1e-6)
    targets = torch.arange(z1.size(0), device=z1.device)
    # enforce that two different views of the same image are pulled together (positive pairs).
    loss_a = torch.nn.functional.cross_entropy(logits, targets)
    loss_b = torch.nn.functional.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


class SelfSupervisedTrainer:
    # This trainer implements the self-supervised pretraining for RGBD+SelfSup baseline model.
    # also the same pretraining strategy used for main model.
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.device = device

        # Only support backbones exposing avgpool
        if not hasattr(model, "avgpool") or not hasattr(model, "fc"):
            raise RuntimeError("Self-supervised pretraining currently supports ResNet/Inception backbones only.")
        self.pool = getattr(model, "avgpool")
        # attach a projection head to the backbone's feature extractor.
        self.proj = ProjectionHead(getattr(model, "fc").in_features).to(device)  # type: ignore[arg-type]
        lr = self.config.selfsup_learning_rate or self.config.learning_rate
        self.optimizer = AdamW(list(model.parameters()) + list(self.proj.parameters()), lr=lr, weight_decay=self.config.weight_decay)
        self.scaler = amp.GradScaler(enabled=self.config.use_amp and torch.cuda.is_available())

    def _extract_pooled(self, x: torch.Tensor) -> torch.Tensor:
        # use a hook to extract global features from the backbone's average pooling layer.
        hook = FeatureHook(self.pool)
        _ = self.model(x)
        pooled = hook.tensor
        hook.close()
        if pooled is None:
            raise RuntimeError("Failed to capture backbone pooled features for self-supervised training.")
        return torch.flatten(pooled, 1)

    def train(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        if self.config.selfsup_epochs <= 0:
            return history
        self.model.train()
        self.proj.train()
        for epoch in range(self.config.selfsup_epochs):
            running = 0.0
            steps = 0
            for x, _ in tqdm(self.dataloader, desc=f"self-sup {epoch+1}/{self.config.selfsup_epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)
                # create two different augmented views of the same input tensor.
                v1 = random_tensor_augment(x)
                v2 = random_tensor_augment(x)
                if self.config.selfsup_depth_mask_ratio > 0.0:
                    # As described, apply random mask occlusion to the depth channel in each view.
                    v1 = apply_depth_patch_mask_on_4ch(v1, self.config.selfsup_depth_mask_ratio, self.config.selfsup_patch_size)
                    v2 = apply_depth_patch_mask_on_4ch(v2, self.config.selfsup_depth_mask_ratio, self.config.selfsup_patch_size)
                with amp.autocast(enabled=self.scaler.is_enabled()):
                    # extract pooled features for both views.
                    h1 = self._extract_pooled(v1)
                    h2 = self._extract_pooled(v2)
                    # project them into the contrastive space.
                    z1 = self.proj(h1)
                    z2 = self.proj(h2)
                    # use an InfoNCE objective to enforce view consistency.
                    loss = info_nce_views(z1, z2, self.config.selfsup_temperature)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running += loss.item()
                steps += 1
            if steps:
                avg = running / steps
                LOGGER.info("Self-supervised epoch %d/%d | contrastive %.4f", epoch + 1, self.config.selfsup_epochs, avg)
                history.append({"epoch": epoch + 1, "contrastive_loss": avg})
        # The goal of this stage is to learn good representations before formal, supervised training.
        return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    target_mean: float,
) -> Dict[str, float]:
    """
    Run evaluation on a dataloader and return averaged metrics.

    Args:
        model: Model to evaluate.
        dataloader: Validation dataloader.
        device: Compute device.
        criterion: Loss function.
        target_mean: Mean of targets for relative error.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[float] = []
    all_targets: List[float] = []
    for images, targets in tqdm(dataloader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds, _, loss = forward_with_optional_aux(
            model, images, targets, criterion, include_aux_loss=False
        )
        if loss is not None:
            running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())
    # evaluate the model using Mean Absolute Error (MAE) and RMSE, consistent with report metrics.
    metrics = compute_metrics(all_preds, all_targets, target_mean)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    return metrics


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> List[Tuple[str, float]]:
    """
    Run inference and return (dish_id, prediction) pairs.

    Args:
        model: Trained model.
        dataloader: Test dataloader yielding images and IDs.
        device: Compute device.
    """
    model.eval()
    results: List[Tuple[str, float]] = []
    for images, dish_ids in tqdm(dataloader, desc="test", leave=False):
        images = images.to(device, non_blocking=True)
        preds, _, _ = forward_with_optional_aux(model, images, None, None, include_aux_loss=False)
        preds = preds.detach().cpu().numpy().tolist()
        results.extend(zip(dish_ids, preds))
    return results


def prepare_dataloaders(config: TrainConfig, input_size: int) -> Tuple[DataLoader, DataLoader, float]:
    """
    Create train/val dataloaders and compute target mean.

    Args:
        config: Training config.
        input_size: Input image size.

    Returns:
        Train loader, val loader, and target mean.
    """
    csv_path = config.data_root / "nutrition5k_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find training CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    all_ids = df["ID"].tolist()
    label_map = dict(zip(df["ID"], df["Value"]))

    rng = random.Random(config.seed)
    rng.shuffle(all_ids)
    # use a 90%/10% random split for train/validation.
    val_size = max(1, int(len(all_ids) * config.val_ratio))
    val_ids = all_ids[:val_size]
    train_ids = all_ids[val_size:]
    if not train_ids:
        raise ValueError("Training split is empty; adjust val_ratio.")

    # initialize the dataset to provide 4-channel (RGB+D) tensors for early-fusion baseline.
    train_dataset = Nutrition5KRGBDDataset(
        train_ids,
        data_root=config.data_root,
        depth_source=config.depth_source,
        input_size=input_size,
        labels=label_map,
        train=True,
    )
    val_dataset = Nutrition5KRGBDDataset(
        val_ids,
        data_root=config.data_root,
        depth_source=config.depth_source,
        input_size=input_size,
        labels=label_map,
        train=True,
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

    train_targets = [label_map[dish_id] for dish_id in train_ids]
    target_mean = float(np.mean(train_targets))
    return train_loader, val_loader, target_mean


def save_checkpoint(
    model: nn.Module,
    optimizer: RMSprop,
    scaler: amp.GradScaler,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
    path: Path,
) -> None:
    """
    Save model, optimizer, scaler, and config to a file.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scaler: AMP scaler state.
        epoch: Current epoch.
        metrics: Metrics to store.
        config: Train configuration.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": metrics,
        "config": asdict(config),
    }
    torch.save(payload, path)

def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[RMSprop] = None, scaler: Optional[amp.GradScaler] = None) -> Dict:
    """
    Load checkpoint and optionally optimizer/scaler states.

    Args:
        path: Checkpoint file path.
        model: Model to load into.
        optimizer: Optional optimizer to load.
        scaler: Optional AMP scaler to load.

    Returns:
        Loaded checkpoint dict.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint

def build_model() -> nn.Module:
    """
    Build a 4-channel ResNet34 regressor and return model and input size.

    Returns:
        Tuple of (model, input_size).
    """
    # builds baseline RGBD model, which is a single ResNet-34 tower.
    input_size = 224
    # select ResNet-34 as the backbone for baseline model.
    model = models.resnet34(weights=None)
    # The backbone output is a linear layer directly regressing calories.
    model.fc = nn.Linear(model.fc.in_features, 1)
    # model = adapt_backbone_for_4ch(model)
    old = model.conv1
    # explicitly replace the first convolutional layer to handle the 4-channel (RGB+D) input.
    new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        # initialize the 4th (depth) channel's weights with the mean of the RGB weights.
        new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.conv1 = new
    return model, input_size

def train_and_evaluate(config: TrainConfig) -> None:
    """
    Main training loop with optional self-supervised warmup.

    Trains, evaluates, logs, and saves checkpoints.
    """
    setup_logging(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_size = build_model()
    model = model.to(device)
    train_loader, val_loader, target_mean = prepare_dataloaders(config, input_size)

    # first run the self-supervised pretraining if specified and not resuming.
    if config.selfsup_epochs > 0 and (config.checkpoint_path is None or not config.checkpoint_path.exists()):
        try:
            ss = SelfSupervisedTrainer(model, config, train_loader, device)
            pre_hist = ss.train()
            # log the history of the self-supervised stage.
            for entry in pre_hist:
                metrics = {k: v for k, v in entry.items() if k != "epoch"}
                append_history_entry(config.output_dir, config.history_filename, {"epoch": entry["epoch"], "phase": "selfsup", "metrics": metrics})
        except RuntimeError as e:
            LOGGER.warning("Self-supervised pretraining skipped: %s", e)

    optimizer = RMSprop(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        alpha=0.9,
        eps=1.0,
        weight_decay=config.weight_decay,
    )
    scaler = amp.GradScaler(enabled=config.use_amp and torch.cuda.is_available())
    # use L1Loss (MAE), which aligns with initial optimization objective (SmoothL1).
    criterion = nn.L1Loss()

    start_epoch = 1
    best_val_mae = float("inf")
    best_checkpoint_path = config.output_dir / f"{config.model_name}_{config.modality}_best.pt"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.checkpoint_path is not None:
        checkpoint = load_checkpoint(config.checkpoint_path, model, optimizer, scaler)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_mae = checkpoint.get("metrics", {}).get("val_mae", best_val_mae)
        LOGGER.info("Resumed from %s (epoch %d)", config.checkpoint_path, start_epoch - 1)

    include_aux_loss = config.model_name == "inception_v3"
    history: List[Dict[str, float]] = []
    # the main supervised training loop.
    for epoch in range(start_epoch, config.epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, config.epochs)
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        all_preds: List[float] = []
        all_targets: List[float] = []
        for step, (images, targets) in enumerate(tqdm(train_loader, desc="train", leave=False), start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=scaler.is_enabled()):
                # get the direct calorie regression from the baseline model.
                preds, _, loss = forward_with_optional_aux(
                    model, images, targets, criterion, include_aux_loss
                )
            if loss is None:
                raise RuntimeError("Loss computation failed during training.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_val = loss.item()
            running_loss += loss_val * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            if config.log_interval and (step % max(1, config.log_interval) == 0):
                LOGGER.info("train step %d/%d | loss %.4f", step, len(train_loader), loss_val)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(all_preds, all_targets, np.mean(all_targets))
        train_metrics["loss"] = epoch_loss
        train_metrics["time_sec"] = float(time.time() - epoch_start)
        try:
            train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])  # type: ignore[index]
        except Exception:
            pass
        LOGGER.info("Train: %s", format_metrics(train_metrics))
        append_history_entry(
            config.output_dir,
            config.history_filename,
            {"epoch": epoch, "phase": "train", "metrics": train_metrics},
        )

        t_val0 = time.time()
        # compute validation metrics, primarily MAE and RMSE, to track performance.
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            target_mean=target_mean,
        )
        val_metrics["time_sec"] = float(time.time() - t_val0)
        try:
            val_metrics["lr"] = float(optimizer.param_groups[0]["lr"])  # type: ignore[index]
        except Exception:
            pass
        val_mae = val_metrics["mae"]
        LOGGER.info("Val  : %s", format_metrics(val_metrics))
        append_history_entry(
            config.output_dir,
            config.history_filename,
            {"epoch": epoch, "phase": "val", "metrics": val_metrics},
        )

        latest_path = config.output_dir / f"{config.model_name}_{config.modality}_latest.pt"
        save_checkpoint(model, optimizer, scaler, epoch, {"val_mae": val_mae}, config, latest_path)
        # save the model if it achieves a new best validation MAE.
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scaler, epoch, {"val_mae": best_val_mae}, config, best_checkpoint_path)
        if epoch % max(1, config.save_interval) == 0:
            tag_path = config.output_dir / f"{config.model_name}_{config.modality}_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scaler, epoch, {"val_mae": val_mae}, config, tag_path)

    history_path = config.output_dir / f"{config.model_name}_{config.modality}_history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2, ensure_ascii=False)
    LOGGER.info("Training history saved to %s", history_path)

    # load the best checkpoint for final evaluation.
    checkpoint = load_checkpoint(best_checkpoint_path, model)
    LOGGER.info("Loaded best checkpoint from epoch %s", checkpoint.get('epoch'))

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion,
        target_mean=target_mean,
    )
    LOGGER.info(
        "Best validation MAE: %.4f, Relative Error: %.2f%%",
        final_val_metrics['mae'], final_val_metrics['rel_err_pct']
    )


def parse_args() -> TrainConfig:
    """
    Parse CLI args and build a TrainConfig.

    Returns:
        A populated TrainConfig object.
    """
    parser = argparse.ArgumentParser(description="RGB-D early-fusion baseline with self-supervised pretraining")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("comp-90086-nutrition-5-k") / "Nutrition5K" / "Nutrition5K",
        help="Root directory containing the Nutrition5K dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="depth_color",
        choices=["depth_color", "depth_raw"],
        help="Depth source for RGB-D early fusion.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader worker processes.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Enable automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.set_defaults(use_amp=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "rgbd_selfsup")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resume or evaluate from an existing checkpoint.",
    )
    parser.add_argument("--log-interval", type=int, default=30, help="Steps between train log messages.")
    parser.add_argument("--history-filename", type=str, default="history.jsonl", help="Per-epoch metrics JSONL name.")
    parser.add_argument("--save-interval", type=int, default=30, help="Save checkpoint every N epochs.")
    # These flags control the self-supervised pretraining stage.
    parser.add_argument("--self-sup-epochs", type=int, default=20, help="Number of self-supervised pretraining epochs.")
    parser.add_argument("--self-sup-temperature", type=float, default=0.2, help="Contrastive temperature.")
    parser.add_argument("--self-sup-lr", type=float, default=None, help="Learning rate for self-supervised stage (defaults to --lr).")
    parser.add_argument("--self-sup-depth-mask-ratio", type=float, default=0.25, help="Mask ratio on depth channel during self-sup.")
    parser.add_argument("--self-sup-patch-size", type=int, default=8, help="Patch size for depth masking in self-sup.")

    args = parser.parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        modality="rgbd",
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        depth_source=args.depth,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        log_interval=args.log_interval,
        history_filename=args.history_filename,
        save_interval=args.save_interval,
        selfsup_epochs=args.selfsup_epochs,
        selfsup_temperature=args.selfsup_temperature,
        selfsup_learning_rate=args.selfsup_lr,
        selfsup_depth_mask_ratio=args.selfsup_depth_mask_ratio,
        selfsup_patch_size=args.selfsup_patch_size,
    )
    return config


def main() -> None:
    """
    Entry point for CLI training.

    Parses args, then runs training and evaluation.
    """
    config = parse_args()
    seed_everything(config.seed)
    train_and_evaluate(config)


if __name__ == "__main__":
    main()