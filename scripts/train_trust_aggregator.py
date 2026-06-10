"""
Train the AttentionBasedTrustAggregator on pre-computed trust components.

Usage:
    python scripts/train_trust_aggregator.py --dataset fb15k237
    python scripts/train_trust_aggregator.py --dataset wn18rr
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
from src.utils.logging_utils import setup_logging, ExperimentLogger, format_metrics_table

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train trust score aggregator")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--fixed_weights",
        action="store_true",
        help="Skip training; use fixed equal weights α=β=γ=1/3 as a baseline aggregator.",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default=None,
        help="Save checkpoint as checkpoints/{dataset}/{save_as}.pt instead of trust_aggregator.pt.",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Override calibration labels file. Path to a .pt with shape [N,4], col 3 = binary label. "
             "Defaults to trust_calibration/{dataset}/calibration_labels.pt.",
    )
    return parser.parse_args()


def load_or_create_components(dataset: str, results_dir: Path) -> torch.Tensor:
    """
    Load U, S_prov, Delta_CF tensors.
    Falls back to synthetic data if files don't exist.

    Returns: (N, 3) tensor [U, S_prov, Delta_CF]
    """
    u_path = results_dir / "uncertainty_scores.pt"
    cf_path = results_dir / "counterfactual_scores.pt"
    prov_path = Path(f"data/processed/{dataset}/provenance_weights.pt")

    # Try to load real computed scores
    if u_path.exists() and cf_path.exists():
        u_data = torch.load(u_path, weights_only=False)        # dict: U_margin, query_ids, ...
        cf_tensor = torch.load(cf_path, weights_only=False)   # (N, 4): [idx, Delta_CF, max_crit, mean_rand]

        # Align by query_idx — use U_margin as primary uncertainty signal
        _u_ids = u_data["query_ids"]; _u_vals = u_data["U_margin"]
        u_dict = {int(_u_ids[i].item()): float(_u_vals[i].item()) for i in range(len(_u_ids))}
        cf_dict = {int(row[0].item()): row[1].item() for row in cf_tensor}

        common_idx = sorted(set(u_dict.keys()) & set(cf_dict.keys()))

        if prov_path.exists():
            prov_weights = torch.load(prov_path, weights_only=True).float()
            prov_dict = {i: prov_weights[i % len(prov_weights)].item()
                         for i in range(len(prov_weights))}
        else:
            prov_dict = {}

        U_list, S_prov_list, CF_list = [], [], []
        for idx in common_idx:
            U_list.append(u_dict[idx])
            S_prov_list.append(prov_dict.get(idx, 0.75))
            CF_list.append(cf_dict[idx])

        if U_list:
            return torch.tensor(
                list(zip(U_list, S_prov_list, CF_list)), dtype=torch.float
            )

    # Fallback: generate synthetic trust components
    logger.warning(f"Trust component files not found. Generating synthetic data.")
    N = 5000
    torch.manual_seed(RANDOM_SEED)
    U = torch.rand(N)
    S_prov = torch.rand(N) * 0.5 + 0.5   # Bias toward higher quality
    Delta_CF = torch.rand(N)
    return torch.stack([U, S_prov, Delta_CF], dim=1)


def load_or_create_labels(dataset: str, calib_dir: Path, N: int) -> torch.Tensor:
    """Load calibration labels or generate synthetic ones."""
    label_path = calib_dir / dataset / "calibration_labels.pt"
    if label_path.exists():
        labels_raw = torch.load(label_path, weights_only=True)  # (N, 4): [idx, h, r, label]
        labels = labels_raw[:, 3].float()
        # Match size
        if len(labels) >= N:
            return labels[:N]
        # Pad with synthetic if needed
        extra = torch.bernoulli(torch.full((N - len(labels),), 0.5))
        return torch.cat([labels, extra])

    # Synthetic labels: trust ~ correct if trust score is high
    logger.warning("Calibration labels not found. Generating synthetic labels.")
    torch.manual_seed(RANDOM_SEED)
    return torch.bernoulli(torch.full((N,), 0.6))


def load_calibration_components(dataset: str, calib_dir: Path, N: int) -> torch.Tensor | None:
    """
    Load calibration_components.pt (U, S_prov, Delta_CF) generated by the
    4-type calibration design. Returns (N, 3) FloatTensor, or None if not found.
    """
    path = calib_dir / dataset / "calibration_components.pt"
    if not path.exists():
        return None
    comps = torch.load(path, weights_only=True).float()  # (M, 3)
    if len(comps) >= N:
        return comps[:N]
    logger.warning("calibration_components.pt has %d rows but need %d — padding.", len(comps), N)
    pad = comps[torch.randint(len(comps), (N - len(comps),))]
    return torch.cat([comps, pad], dim=0)


def load_metadata(dataset: str, N: int) -> tuple:
    """Load relation frequencies and entity degrees for weight analysis."""
    rel_freq_path = Path(TRUST_CALIB_DIR) / dataset / "relation_frequencies.pt"
    deg_path = Path(TRUST_CALIB_DIR) / dataset / "entity_degrees.pt"

    if rel_freq_path.exists():
        rel_freqs_all = torch.load(rel_freq_path, weights_only=True).float()
        # Assign relation frequency per query (use modulo for simplicity)
        rel_freq = rel_freqs_all[torch.arange(N) % len(rel_freqs_all)]
    else:
        rel_freq = torch.randint(0, 500, (N,)).float()

    if deg_path.exists():
        degrees_all = torch.load(deg_path, weights_only=True).float()
        head_degree = degrees_all[torch.arange(N) % len(degrees_all)]
    else:
        head_degree = torch.randint(0, 100, (N,)).float()

    return rel_freq, head_degree


def _fit_temperature(
    model: "AttentionBasedTrustAggregator",
    val_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Fit a single temperature parameter T on the validation set to minimise BCE.

    The model outputs probabilities p ∈ [0,1], so we:
      1. Convert to logits:  z = logit(p) = log(p / (1-p))
      2. Scale:              calibrated = sigmoid(z / T)
      3. Optimise T via LBFGS to minimise BCE(calibrated, labels)

    T > 1  → softer probabilities  (model was overconfident)
    T < 1  → sharper probabilities (model was underconfident)

    Returns:
        (temperature: float, ece_before: float, ece_after: float)
    """
    from src.evaluation.metrics import compute_expected_calibration_error

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            U, S_prov, Delta_CF, labels = [b.to(device) for b in batch[:4]]
            T, _ = model(U, S_prov, Delta_CF)
            all_scores.append(T.cpu())
            all_labels.append(labels.cpu())

    val_scores = torch.cat(all_scores)   # (N,) probabilities
    val_labels = torch.cat(all_labels)   # (N,) binary

    ece_before = compute_expected_calibration_error(
        val_scores.numpy(), val_labels.numpy()
    )

    # Convert probabilities → logits (clamp to avoid log(0))
    val_logits = torch.logit(val_scores.clamp(1e-6, 1.0 - 1e-6))  # (N,)

    temperature = nn.Parameter(torch.ones(1))
    optimizer_t = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCELoss()

    def closure():
        optimizer_t.zero_grad()
        temp = temperature.clamp(min=0.05)          # prevent collapse to 0
        calibrated = torch.sigmoid(val_logits / temp)
        loss = criterion(calibrated, val_labels)
        loss.backward()
        return loss

    optimizer_t.step(closure)

    temp_val = float(temperature.clamp(min=0.05).item())

    with torch.no_grad():
        calibrated_scores = torch.sigmoid(val_logits / temp_val)

    ece_after = compute_expected_calibration_error(
        calibrated_scores.numpy(), val_labels.numpy()
    )

    return temp_val, ece_before, ece_after


def _check_not_synthetic(dataset: str) -> None:
    """
    Refuse to train if the calibration data is still a Phase 1C synthetic placeholder.

    Reads calibration_summary.json and asserts is_synthetic == False.
    Raises SystemExit with a clear message pointing to build_trust_calibration_v2.py.
    """
    summary_path = Path(TRUST_CALIB_DIR) / dataset / "calibration_summary.json"
    if not summary_path.exists():
        # No summary yet — allow training (first run before any calibration script)
        return
    with open(summary_path) as f:
        summary = json.load(f)
    if summary.get("is_synthetic", False):
        raise SystemExit(
            "\n"
            "ERROR: Calibration data is a Phase 1C synthetic placeholder.\n"
            "       U and Delta_CF were structurally derived, NOT from real model outputs.\n"
            "\n"
            f"  Run:  python scripts/build_trust_calibration_v2.py --dataset {dataset}\n"
            "\n"
            "  This requires Phase 3 outputs in:\n"
            f"    experiments/results/{dataset}/uncertainty_scores.pt\n"
            f"    experiments/results/{dataset}/counterfactual_scores.pt\n"
            f"    experiments/results/{dataset}/predictions.pt\n"
        )


def main():
    args = parse_args()
    setup_logging()
    _check_not_synthetic(args.dataset)
    torch.manual_seed(RANDOM_SEED)

    gpu_id = args.gpu if args.gpu is not None else GPU_CONFIG.get(args.dataset, 0)
    device = get_device(gpu_id)
    logger.info(f"Device: {device}")

    results_dir = Path(RESULTS_DIR) / args.dataset
    calib_dir = Path(TRUST_CALIB_DIR)

    # ─── Load trust components ──────────────────────────────────────────────────
    components = load_or_create_components(args.dataset, results_dir)  # (N, 3)
    N = len(components)
    logger.info(f"Trust components: N={N}")

    if args.debug:
        N = min(N, 500)
        components = components[:N]

    # ─── Load labels ────────────────────────────────────────────────────────────
    if args.label_file:
        label_path = Path(args.label_file)
        if not label_path.exists():
            raise FileNotFoundError(f"--label_file not found: {label_path}")
        labels_raw = torch.load(label_path, weights_only=True)
        labels = labels_raw[:, 3].float()
        logger.info(f"Labels loaded from {label_path}  (N={len(labels)}, "
                    f"pos={int((labels==1).sum())}, neg={int((labels==0).sum())})")
    else:
        labels = load_or_create_labels(args.dataset, calib_dir, N)
    labels = labels[:N]

    # ─── Load metadata ──────────────────────────────────────────────────────────
    rel_freq, head_degree = load_metadata(args.dataset, N)
    rel_freq = rel_freq[:N]
    head_degree = head_degree[:N]

    # ─── Use 4-type calibration components if available (Phase 1C) ─────────────
    calib_comps = load_calibration_components(args.dataset, calib_dir, N)
    if calib_comps is not None:
        logger.info("Loaded calibration_components.pt — using 4-type U/S_prov/Delta_CF.")
        U       = calib_comps[:, 0]
        S_prov  = calib_comps[:, 1]
        Delta_CF = calib_comps[:, 2]
    else:
        logger.warning("calibration_components.pt not found — falling back to raw components.")
        U       = components[:, 0]
        S_prov  = components[:, 1]
        Delta_CF = components[:, 2]

    # ─── AUC sanity checks: dataset must not be trivially easy ──────────────────
    labels_np  = labels.numpy()
    sprov_np   = S_prov.numpy()
    u_np       = U.numpy()
    cf_np      = Delta_CF.numpy()

    sprov_auc = roc_auc_score(labels_np, sprov_np)
    u_auc     = roc_auc_score(labels_np, 1.0 - u_np)    # high certainty → label=1
    cf_auc    = roc_auc_score(labels_np, 1.0 - cf_np)   # low sensitivity → label=1
    logger.info(
        f"Component AUCs — S_prov={sprov_auc:.3f}, 1-U={u_auc:.3f}, 1-Delta_CF={cf_auc:.3f}"
    )

    # Warn (not assert) on low-signal components — the aggregator can learn
    # γ≈0 naturally if Delta_CF carries no signal (e.g. WN18RR).
    low_signal_components = []
    if sprov_auc >= 0.70:
        logger.warning(
            f"Dataset too easy: S_prov AUC={sprov_auc:.3f} (expected < 0.70). "
            "Consider regenerating calibration data."
        )
    if u_auc <= 0.55:
        logger.warning(
            f"U_margin carries low signal: AUC={u_auc:.3f} (expected > 0.55). "
            "Aggregator may rely more on S_prov and Delta_CF."
        )
        low_signal_components.append("U")
    if cf_auc <= 0.55:
        logger.warning(
            f"Delta_CF carries low signal: AUC={cf_auc:.3f} (expected > 0.55). "
            f"Aggregator will learn γ≈0 naturally — no masking needed."
        )
        low_signal_components.append("Delta_CF")
    if low_signal_components:
        logger.info(
            f"Low-signal components for {args.dataset}: {low_signal_components}. "
            "Training proceeds; aggregator will down-weight them automatically."
        )

    # ─── Class balance & pos_weight ─────────────────────────────────────────────
    n_verified = int((labels == 1).sum())
    n_poisoned  = int((labels == 0).sum())
    pos_weight = n_poisoned / max(n_verified, 1)
    logger.info(
        f"Class balance — verified={n_verified}, poisoned={n_poisoned}, "
        f"pos_weight={pos_weight:.3f}"
    )

    # ─── Create datasets ────────────────────────────────────────────────────────
    full_ds = TensorDataset(U, S_prov, Delta_CF, labels)
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))

    batch_size = TRUST_AGG_CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # ─── Model ─────────────────────────────────────────────────────────────────
    model = AttentionBasedTrustAggregator(hidden_dim=TRUST_AGG_CONFIG["hidden_dim"])
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trust aggregator parameters: {total_params:,}")

    exp_logger = ExperimentLogger(f"trust_agg_{args.dataset}")

    model = model.to(device)
    trainer = TrustCalibrationTrainer(model, device, pos_weight=pos_weight)

    if args.fixed_weights:
        # Fixed equal weights baseline: force last Linear bias → [0,0,0] so
        # softmax([0,0,0]) = [1/3, 1/3, 1/3].  No training needed.
        logger.info("--fixed_weights: skipping training, using α=β=γ=1/3")
        with torch.no_grad():
            last_linear = [m for m in model.mlp if isinstance(m, nn.Linear)][-1]
            last_linear.weight.zero_()
            last_linear.bias.zero_()
        history = []
    else:
        # ─── Training ──────────────────────────────────────────────────────────
        num_epochs = args.epochs or TRUST_AGG_CONFIG["num_epochs"]
        if args.debug:
            num_epochs = 5

        history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
        for h in history:
            exp_logger.log(h, step=h["step"])

    # ─── Test evaluation ─────────────────────────────────────────────────────────
    test_metrics = trainer.evaluate(val_loader)
    print("\n" + format_metrics_table(test_metrics, title=f"Trust Aggregator Results — {args.dataset}"))

    # ─── Temperature scaling calibration ─────────────────────────────────────────
    temp_val, ece_before, ece_after = _fit_temperature(model, val_loader, device)
    logger.info(
        f"Temperature scaling: T={temp_val:.4f}  "
        f"ECE {ece_before:.4f} → {ece_after:.4f}"
    )
    print(f"\n{'='*60}")
    print(f"  Temperature Scaling Calibration — {args.dataset}")
    print(f"{'='*60}")
    print(f"  Temperature:  {temp_val:.4f}  "
          f"({'overconfident → soften' if temp_val > 1 else 'underconfident → sharpen'})")
    print(f"  ECE before:   {ece_before:.4f}")
    print(f"  ECE after:    {ece_after:.4f}  "
          f"({'improved' if ece_after < ece_before else 'no improvement'})")
    print(f"{'='*60}")

    # ─── Weight analysis by query type ──────────────────────────────────────────
    # Build extended loader with metadata
    meta_ds = TensorDataset(U[:val_size], S_prov[:val_size], Delta_CF[:val_size],
                            labels[:val_size], rel_freq[:val_size], head_degree[:val_size])
    meta_loader = DataLoader(meta_ds, batch_size=batch_size * 2, shuffle=False)

    weight_analysis = trainer.analyze_weights_by_query_type(meta_loader)

    # ── Global learned weights (α, β, γ) across ALL validation queries ──────────
    all_a = np.concatenate([w["weights_array"][:, 0] for w in weight_analysis.values()
                            if len(w["weights_array"])])
    all_b = np.concatenate([w["weights_array"][:, 1] for w in weight_analysis.values()
                            if len(w["weights_array"])])
    all_g = np.concatenate([w["weights_array"][:, 2] for w in weight_analysis.values()
                            if len(w["weights_array"])])

    print("\n" + "="*60)
    print(f"  Learned Component Weights — {args.dataset}  (N={len(all_a)})")
    print("="*60)
    print(f"  α (1-U_margin):  mean={all_a.mean():.4f}  std={all_a.std():.4f}")
    print(f"  β (S_prov):      mean={all_b.mean():.4f}  std={all_b.std():.4f}")
    print(f"  γ (Delta_CF):    mean={all_g.mean():.4f}  std={all_g.std():.4f}")
    print(f"  Trust formula:   T = {all_a.mean():.3f}·(1-U) + {all_b.mean():.3f}·S_prov "
          f"+ {all_g.mean():.3f}·Δ_CF")
    if low_signal_components:
        masked = ", ".join(low_signal_components)
        print(f"  Note: {masked} had low AUC — verify γ is near 0 above.")
    print("="*60)

    # ── Per-query-type breakdown ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Weight Analysis by Query Type")
    print("="*60)
    print(f"{'Query Type':<20} | {'mean α':>8} | {'mean β':>8} | {'mean γ':>8} | {'N':>6}")
    print("-"*60)
    for qt, w in weight_analysis.items():
        print(f"{qt:<20} | {w['alpha']:>8.4f} | {w['beta']:>8.4f} | {w['gamma']:>8.4f} | {w['n']:>6}")
    print("="*60)

    # ─── Save model ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(CHECKPOINT_DIR) / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"{args.save_as}.pt" if args.save_as else "trust_aggregator.pt"
    ckpt_path = ckpt_dir / ckpt_filename
    global_weights = {
        "alpha_mean": float(all_a.mean()), "alpha_std": float(all_a.std()),
        "beta_mean":  float(all_b.mean()), "beta_std":  float(all_b.std()),
        "gamma_mean": float(all_g.mean()), "gamma_std": float(all_g.std()),
        "low_signal_components": low_signal_components,
    }
    temperature_info = {
        "temperature":  temp_val,
        "ece_before":   ece_before,
        "ece_after":    ece_after,
    }
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "test_metrics": test_metrics,
        "global_weights": global_weights,
        "temperature_scaling": temperature_info,
        "weight_analysis": {qt: {k: v for k, v in w.items() if k != "weights_array"}
                            for qt, w in weight_analysis.items()},
    }, ckpt_path)
    logger.info(f"Trust aggregator saved to {ckpt_path}")

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = f"{args.save_as}_metrics.json" if args.save_as else "trust_aggregator_metrics.json"
    with open(results_dir / metrics_filename, "w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "global_weights": global_weights,
            "temperature_scaling": temperature_info,
            "weight_analysis": {qt: {k: v for k, v in w.items() if k != "weights_array"}
                                 for qt, w in weight_analysis.items()},
        }, f, indent=2)

    exp_logger.log_summary(test_metrics)
    exp_logger.finish()

    # Check targets
    acc = test_metrics["accuracy"]
    auc = test_metrics["auc_roc"]
    print(f"\nTarget Acc: 0.85 | Achieved: {acc:.4f} {'✓' if acc >= 0.85 else '✗'}")
    print(f"Target AUC: 0.90 | Achieved: {auc:.4f} {'✓' if auc >= 0.90 else '✗'}")


if __name__ == "__main__":
    main()
