"""
Ablation training: zero out one trust component and retrain aggregator.

Usage:
    python scripts/ablation_train.py --dataset fb15k237 --zero_component 0  # no_uncertainty
    python scripts/ablation_train.py --dataset fb15k237 --zero_component 1  # no_provenance
    python scripts/ablation_train.py --dataset fb15k237 --zero_component 2  # no_counterfactual

Component indices:
    0 → 1-U        (uncertainty)
    1 → S_prov     (provenance)
    2 → Delta_CF   (counterfactual)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TRUST_AGG_CONFIG, CHECKPOINT_DIR, RESULTS_DIR,
    TRUST_CALIB_DIR, GPU_CONFIG, RANDOM_SEED
)
from src.models.trust.aggregator import AttentionBasedTrustAggregator, TrustCalibrationTrainer
from src.utils.gpu_utils import get_device
from src.utils.logging_utils import setup_logging
from src.evaluation.metrics import compute_expected_calibration_error

logger = logging.getLogger(__name__)

COMPONENT_NAMES = {0: "no_uncertainty", 1: "no_provenance", 2: "no_counterfactual"}
COMPONENT_LABELS = {0: "1-U (uncertainty)", 1: "S_prov (provenance)", 2: "Delta_CF (counterfactual)"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train ablation trust aggregator")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument(
        "--zero_component", required=True, type=int, choices=[0, 1, 2],
        help="Column index to zero out: 0=1-U, 1=S_prov, 2=Delta_CF",
    )
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _fit_temperature(model, val_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            U, S_prov, Delta_CF, labels = [b.to(device) for b in batch[:4]]
            T, _ = model(U, S_prov, Delta_CF)
            all_scores.append(T.cpu())
            all_labels.append(labels.cpu())

    val_scores = torch.cat(all_scores)
    val_labels = torch.cat(all_labels)

    ece_before = compute_expected_calibration_error(val_scores.numpy(), val_labels.numpy())

    val_logits = torch.logit(val_scores.clamp(1e-6, 1.0 - 1e-6))
    temperature = nn.Parameter(torch.ones(1))
    optimizer_t = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCELoss()

    def closure():
        optimizer_t.zero_grad()
        temp = temperature.clamp(min=0.05)
        calibrated = torch.sigmoid(val_logits / temp)
        loss = criterion(calibrated, val_labels)
        loss.backward()
        return loss

    optimizer_t.step(closure)
    temp_val = float(temperature.clamp(min=0.05).item())

    with torch.no_grad():
        calibrated_scores = torch.sigmoid(val_logits / temp_val)
    ece_after = compute_expected_calibration_error(calibrated_scores.numpy(), val_labels.numpy())

    return temp_val, ece_before, ece_after


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(RANDOM_SEED)

    ablation_name = COMPONENT_NAMES[args.zero_component]
    zeroed_label = COMPONENT_LABELS[args.zero_component]
    logger.info(f"Ablation: {ablation_name} — zeroing column {args.zero_component} ({zeroed_label})")

    gpu_id = args.gpu if args.gpu is not None else GPU_CONFIG.get(args.dataset, 0)
    device = get_device(gpu_id)

    calib_dir = Path(TRUST_CALIB_DIR)
    comp_path = calib_dir / args.dataset / "calibration_components.pt"
    label_path = calib_dir / args.dataset / "calibration_labels.pt"

    if not comp_path.exists():
        raise FileNotFoundError(f"calibration_components.pt not found at {comp_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"calibration_labels.pt not found at {label_path}")

    # Load components and labels (do NOT modify originals)
    comps = torch.load(comp_path, weights_only=True).float().clone()  # (N, 3)
    labels_raw = torch.load(label_path, weights_only=True)
    labels = labels_raw[:, 3].float()

    N = min(len(comps), len(labels))
    comps = comps[:N]
    labels = labels[:N]

    # Zero out the ablated column
    comps[:, args.zero_component] = 0.0
    logger.info(f"Zeroed column {args.zero_component} ({zeroed_label}) in {N} samples")

    U        = comps[:, 0]
    S_prov   = comps[:, 1]
    Delta_CF = comps[:, 2]

    # Class balance
    n_verified = int((labels == 1).sum())
    n_poisoned  = int((labels == 0).sum())
    pos_weight = n_poisoned / max(n_verified, 1)
    logger.info(f"Class balance — verified={n_verified}, poisoned={n_poisoned}, pos_weight={pos_weight:.3f}")

    # Train/val split
    full_ds = TensorDataset(U, S_prov, Delta_CF, labels)
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    batch_size = TRUST_AGG_CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False)

    # Model + training
    model = AttentionBasedTrustAggregator(hidden_dim=TRUST_AGG_CONFIG["hidden_dim"])
    model = model.to(device)
    trainer = TrustCalibrationTrainer(model, device, pos_weight=pos_weight)

    num_epochs = TRUST_AGG_CONFIG["num_epochs"]
    logger.info(f"Training for {num_epochs} epochs...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Evaluate
    test_metrics = trainer.evaluate(val_loader)
    logger.info(f"Val metrics: {test_metrics}")

    # Temperature scaling
    temp_val, ece_before, ece_after = _fit_temperature(model, val_loader, device)
    logger.info(f"Temperature: {temp_val:.4f}  ECE {ece_before:.4f} → {ece_after:.4f}")

    # Save checkpoint
    ckpt_dir = Path(CHECKPOINT_DIR) / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"trust_aggregator_{ablation_name}.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "test_metrics": test_metrics,
        "ablation": ablation_name,
        "zeroed_column": args.zero_component,
        "temperature_scaling": {
            "temperature":  temp_val,
            "ece_before":   ece_before,
            "ece_after":    ece_after,
        },
    }, ckpt_path)
    logger.info(f"Saved ablation checkpoint: {ckpt_path}")

    print(f"\n{'='*55}")
    print(f"  Ablation: {ablation_name}")
    print(f"  AUC-ROC : {test_metrics['auc_roc']:.4f}")
    print(f"  ECE     : {ece_after:.4f}  (after temperature scaling)")
    print(f"  Ckpt    : {ckpt_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
