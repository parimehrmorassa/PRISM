"""
build_trust_calibration_v2.py

Builds trust calibration data from REAL Phase 3 model outputs.

Unlike create_trust_calibration.py (which uses structurally-derived synthetic
U and Delta_CF values), this script uses the actual scores produced by the
conformal uncertainty, counterfactual, and provenance modules.

Prerequisites (must have run Phase 3 first):
    experiments/results/{dataset}/uncertainty_scores.pt
    experiments/results/{dataset}/counterfactual_scores.pt
    data/processed/{dataset}/provenance_weights.pt
    experiments/results/{dataset}/predictions.pt

File formats (from Phase 3):
    uncertainty_scores.pt     — (N, 3) float32: [query_idx, U, pred_set_size]
    counterfactual_scores.pt  — (N, 4) float32: [query_idx, Delta_CF, max_crit, mean_rand]
    provenance_weights.pt     — (num_train_edges,) float32: one weight per training edge
    predictions.pt            — (N, 3) long: [query_idx, true_tail_id, predicted_tail_id]

Output (written to TRUST_CALIB_DIR/{dataset}/, overwrites Phase 1C synthetic data):
    calibration_labels.pt      — (N, 4) LongTensor  [query_idx, head_id, rel_id, label]
    calibration_components.pt  — (N, 3) FloatTensor [U, S_prov, Delta_CF]
    calibration_summary.json   — metadata with "is_synthetic": False

Usage:
    python scripts/build_trust_calibration_v2.py --dataset fb15k237
    python scripts/build_trust_calibration_v2.py --dataset wn18rr --gpu 0
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RESULTS_DIR,
    TRUST_CALIB_DIR,
    PROCESSED_DATA,
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_uncertainty(results_dir: Path) -> Dict[int, float]:
    """
    Load uncertainty_scores.pt → {query_idx: U_margin}.

    Format: dict with keys U_conformal, U_margin, U_entropy, query_ids.
    U_margin is used as the primary uncertainty signal.
    """
    path = results_dir / "uncertainty_scores.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"uncertainty_scores.pt not found at {path}\n"
            "Run Phase 3 (compute_uncertainty.py) first."
        )
    data = torch.load(path, weights_only=False)
    query_ids = data["query_ids"]
    u_vals = data["U_margin"]
    return {int(query_ids[i].item()): float(u_vals[i].item()) for i in range(len(query_ids))}


def _load_counterfactual(results_dir: Path) -> Dict[int, float]:
    """
    Load counterfactual_scores.pt → {query_idx: Delta_CF}.

    Format: (N, 4) float32 — [query_idx, Delta_CF, max_crit, mean_rand]
    """
    path = results_dir / "counterfactual_scores.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"counterfactual_scores.pt not found at {path}\n"
            "Run Phase 3 (compute_counterfactuals.py) first."
        )
    tensor = torch.load(path, weights_only=False)  # (N, 4): [query_idx, Delta_CF, max_crit, mean_rand]
    return {int(row[0].item()): float(row[1].item()) for row in tensor}


def _load_provenance(dataset: str) -> Optional[torch.Tensor]:
    """
    Load provenance_weights.pt.

    Format: (num_train_edges,) float32 — one weight per training edge.
    Returns None if file not found (will fall back to uniform 0.75).
    """
    path = Path(PROCESSED_DATA[dataset]) / "provenance_weights.pt"
    if not path.exists():
        log.warning(
            "provenance_weights.pt not found at %s — using S_prov=0.75 for all queries.", path
        )
        return None
    return torch.load(path, weights_only=True).float()


def _load_test_ranks(results_dir: Path) -> Optional[Dict[int, int]]:
    """
    Load test_ranks.pt → {query_idx: filtered_rank}.

    Format: (N, 2) LongTensor — [query_idx, filtered_rank]
    filtered_rank uses the same filtering as evaluate(): other known true answers
    for the same (head, relation) pair are excluded before computing rank.

    Returns None if the file does not exist.
    """
    path = results_dir / "test_ranks.pt"
    if not path.exists():
        return None
    tensor = torch.load(path, weights_only=False)  # (N, 2)
    return {int(row[0].item()): int(row[1].item()) for row in tensor}


def _compute_filtered_ranks(dataset: str, results_dir: Path) -> Dict[int, int]:
    """
    Compute per-query filtered ranks by re-running the trained NBFNet model.

    For each test query (h, r, t):
      1. Extract subgraph around h
      2. Score all subgraph nodes
      3. Skip nodes that are other known true answers for (h, r)
      4. Record the 1-indexed rank of t among remaining nodes

    This matches the filtering used in trainer.evaluate() and therefore aligns
    with the reported Hits@10 metric.

    Saves test_ranks.pt alongside predictions.pt for future runs.
    Takes ~30 minutes for Hetionet on a single GPU.
    """
    from config import NBFNET_CONFIG, GPU_CONFIG, CHECKPOINT_DIR
    from src.data.dataset import build_dataloaders
    from src.models.nbfnet.model import NBFNet
    from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
    from src.utils.gpu_utils import get_device, autocast_ctx, move_batch_to_device, get_amp_dtype
    from tqdm import tqdm

    gpu_id = GPU_CONFIG.get(dataset, 0)
    device = get_device(gpu_id)
    log.info("Computing filtered ranks on %s (this takes ~30 min for Hetionet) ...", device)

    _, _, test_loader, train_ds = build_dataloaders(
        dataset, batch_size=64, num_workers=4, device="cpu"
    )
    num_entities = train_ds.num_entities

    ckpt_path = Path(CHECKPOINT_DIR) / dataset / "nbfnet_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train_nbfnet.py first to produce a trained model."
        )
    model = NBFNet.load(str(ckpt_path), device=device)
    model.eval()
    log.info("  Loaded checkpoint: %s", ckpt_path)

    all_triples_tensor = train_ds.get_all_triples_tensor()
    all_edge_index = torch.stack([all_triples_tensor[:, 0], all_triples_tensor[:, 2]], dim=0)
    all_edge_type  = all_triples_tensor[:, 1]
    all_edge_prov  = train_ds.get_all_provenance()

    trainer = FullGraphNBFNetTrainer(
        model=model, device=device, dataset_name=dataset,
        all_triples=train_ds.true_triples, num_entities=num_entities,
        all_edge_index=all_edge_index, all_edge_type=all_edge_type,
        all_edge_prov=all_edge_prov,
    )
    trainer.use_amp = NBFNET_CONFIG.get("use_amp", False) and device.type == "cuda"
    amp_dtype = get_amp_dtype(NBFNET_CONFIG.get("amp_dtype", "float16"))

    rows = []  # [(query_idx, filtered_rank), ...]
    query_idx_counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing filtered ranks"):
            batch = move_batch_to_device(batch, "cpu")
            heads = batch["head"]
            rels  = batch["relation"]
            tails = batch["tail"]

            for i in range(len(heads)):
                h, r, t = heads[i].item(), rels[i].item(), tails[i].item()

                subgraph = trainer._extract_subgraph(h, r, t, eval_mode=True)
                if subgraph is None:
                    rows.append((query_idx_counter, num_entities))  # worst-case rank
                    query_idx_counter += 1
                    continue

                edge_index = subgraph["edge_index"].to(device)
                edge_type  = subgraph["edge_type"].to(device)
                edge_prov  = subgraph["edge_prov"].to(device)
                local_head = subgraph["local_head"]
                local_tail = subgraph["local_tail"]
                num_nodes  = subgraph["num_nodes"]
                g2l        = subgraph["global_to_local"]
                local_to_global = {v: k for k, v in g2l.items()}

                with autocast_ctx(device, trainer.use_amp, amp_dtype):
                    scores, _ = trainer.model(
                        edge_index=edge_index, edge_type=edge_type, edge_prov=edge_prov,
                        query_head=local_head, query_relation=r, num_nodes=num_nodes,
                    )

                # Filtered rank: skip other known true answers for (h, r)
                true_score = scores[local_tail].item()
                rank = 1
                for node_local_idx, node_score in enumerate(scores):
                    if node_local_idx == local_tail:
                        continue
                    global_id = local_to_global.get(node_local_idx, -1)
                    if global_id == -1:
                        continue
                    if (h, r, global_id) in trainer.all_triples and global_id != t:
                        continue  # filtered out
                    if node_score.item() > true_score:
                        rank += 1

                rows.append((query_idx_counter, rank))
                query_idx_counter += 1

    test_ranks_tensor = torch.tensor(rows, dtype=torch.long)  # (N, 2)
    save_path = results_dir / "test_ranks.pt"
    torch.save(test_ranks_tensor, str(save_path))

    N = len(test_ranks_tensor)
    n_hits10 = int((test_ranks_tensor[:, 1].clamp(min=1) <= 10).sum().item())
    log.info("test_ranks.pt saved: N=%d, filtered hits@10=%.4f", N, n_hits10 / max(N, 1))

    return {int(row[0].item()): int(row[1].item()) for row in test_ranks_tensor}


def _load_predictions(results_dir: Path) -> Dict[int, Tuple[int, int, int]]:
    """
    Load predictions.pt → {query_idx: (true_tail_id, predicted_tail_id, true_tail_rank)}.

    Supports both formats:
      (N, 3) long — [query_idx, true_tail_id, predicted_tail_id]          (legacy)
      (N, 4) long — [query_idx, true_tail_id, predicted_tail_id, rank]    (current)

    true_tail_rank is -1 when unavailable (legacy format or failed subgraph).
    Returns empty dict if the file does not exist.
    """
    path = results_dir / "predictions.pt"
    if not path.exists():
        return {}
    tensor = torch.load(path, weights_only=False)  # (N, 3) or (N, 4)
    has_rank = tensor.shape[1] >= 4
    return {
        int(row[0].item()): (
            int(row[1].item()),
            int(row[2].item()),
            int(row[3].item()) if has_rank else -1,
        )
        for row in tensor
    }


def _load_test_triples(dataset: str) -> torch.Tensor:
    """
    Load test triples as (N, 3) LongTensor [head, relation, tail].
    Used to associate query indices with head/relation IDs.
    """
    path = Path(PROCESSED_DATA[dataset]) / "triples_test.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"triples_test.pt not found at {path}\n"
            "Run download_datasets.py first."
        )
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# S_prov assignment
# ---------------------------------------------------------------------------

def _assign_sprov(
    query_indices: list,
    prov_weights: Optional[torch.Tensor],
    test_triples: torch.Tensor,
    dataset: str,
    results_dir: Path,
) -> Dict[int, float]:
    """
    Assign S_prov per query.

    Strategy:
      1. If provenance_weights.pt exists, map each query_idx → head entity's
         average provenance weight (using training edge positions as proxy).
         Normalise to [0, 1].
      2. Otherwise fall back to 0.75 for all queries.

    Note: provenance_weights.pt is indexed by training edge position, not by
    entity id. We use (query_idx % num_prov_weights) as a lightweight proxy
    until a proper edge-to-query alignment is available from Phase 3.
    """
    if prov_weights is None:
        return {idx: 0.75 for idx in query_indices}

    # Normalise provenance weights to [0, 1]
    p_min, p_max = prov_weights.min().item(), prov_weights.max().item()
    if p_max > p_min:
        prov_norm = (prov_weights - p_min) / (p_max - p_min)
    else:
        prov_norm = torch.full_like(prov_weights, 0.75)

    n_prov = len(prov_norm)
    result = {}
    for idx in query_indices:
        result[idx] = float(prov_norm[idx % n_prov].item())
    return result


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _assign_labels(
    query_indices: list,
    pred_dict: Dict[int, Tuple[int, int, int]],
    dataset: str,
    rank_dict: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """
    Assign binary labels per query.

    Label strategy (dataset-specific):
      hetionet  — label = 1 if filtered_rank <= 10
                  Uses rank_dict (test_ranks.pt) when available — these are filtered
                  ranks that match the reported Hits@10 metric (~40% positive rate).
                  Falls back to unfiltered rank from predictions.pt col 3 if missing.
      all others — label = 1 if predicted_tail_id == true_tail_id (top-1)
                  (~9-12% positive rate — already meaningful)

    Args:
        rank_dict: {query_idx: filtered_rank} from test_ranks.pt (hetionet only).
                   When provided, takes priority over predictions.pt col 3.

    Raises RuntimeError if predictions.pt is missing entirely.
    """
    if not pred_dict and rank_dict is None:
        raise RuntimeError(
            "predictions.pt not found in the results directory.\n"
            "This file must be produced by Phase 3 (train_nbfnet.py or\n"
            "a script that saves model predictions alongside true tail IDs).\n"
            "Format required: (N, 4) LongTensor "
            "[query_idx, true_tail_id, predicted_tail_id, filtered_rank]"
        )

    use_hits10 = (dataset == "hetionet")
    labels = {}
    n_missing = 0
    n_used_filtered = 0
    n_used_unfiltered = 0

    for idx in query_indices:
        if use_hits10:
            if rank_dict is not None and idx in rank_dict:
                # Filtered rank from test_ranks.pt — matches Hits@10 metric
                labels[idx] = 1 if rank_dict[idx] <= 10 else 0
                n_used_filtered += 1
            elif idx in pred_dict:
                _, pred_tail, rank = pred_dict[idx]
                if rank > 0:
                    # Unfiltered rank from predictions.pt col 3 (fallback)
                    labels[idx] = 1 if rank <= 10 else 0
                    n_used_unfiltered += 1
                else:
                    labels[idx] = 0
                    n_missing += 1
            else:
                labels[idx] = 0
                n_missing += 1
        else:
            if idx in pred_dict:
                _, pred_tail, _ = pred_dict[idx]
                true_tail = pred_dict[idx][0]
                labels[idx] = 1 if pred_tail == true_tail else 0
            else:
                labels[idx] = 0
                n_missing += 1

    if use_hits10:
        if n_used_filtered > 0:
            log.info(
                "hetionet labels: %d queries used filtered rank (test_ranks.pt) "
                "— matches reported Hits@10.",
                n_used_filtered,
            )
        if n_used_unfiltered > 0:
            log.warning(
                "hetionet: %d queries fell back to unfiltered rank from predictions.pt. "
                "Positive rate will be lower than Hits@10. "
                "Run build_trust_calibration_v2.py again to trigger _compute_filtered_ranks().",
                n_used_unfiltered,
            )
    if n_missing > 0:
        log.warning(
            "%d queries had no rank data — defaulted to label=0.",
            n_missing,
        )
    return labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dataset(
    labels: torch.Tensor,
    U: torch.Tensor,
    S_prov: torch.Tensor,
    Delta_CF: torch.Tensor,
) -> None:
    """
    Run AUC sanity checks to verify the dataset is not trivially easy.

    Thresholds (looser than Phase 1C because real data is noisier):
        S_prov alone AUC < 0.80   (should not be trivially predictive)
        1-U AUC > 0.52            (uncertainty must carry some signal)
        1-Delta_CF AUC > 0.52     (counterfactual must carry some signal)
    """
    labels_np  = labels.numpy().astype(int)
    sprov_np   = S_prov.numpy()
    u_np       = U.numpy()
    cf_np      = Delta_CF.numpy()

    if len(set(labels_np)) < 2:
        log.warning("Only one class in labels — AUC undefined. Check predictions.pt.")
        return

    sprov_auc = roc_auc_score(labels_np, sprov_np)
    u_auc     = roc_auc_score(labels_np, 1.0 - u_np)
    cf_auc    = roc_auc_score(labels_np, 1.0 - cf_np)

    log.info(
        "AUC sanity — S_prov=%.3f | 1-U=%.3f | 1-Delta_CF=%.3f",
        sprov_auc, u_auc, cf_auc,
    )
    print(f"  AUC: S_prov={sprov_auc:.3f}  1-U={u_auc:.3f}  1-ΔCF={cf_auc:.3f}")

    if sprov_auc >= 0.80:
        log.warning(
            "S_prov alone achieves AUC=%.3f ≥ 0.80. "
            "The model may be trivially easy for the trust aggregator. "
            "Consider checking provenance weight alignment.",
            sprov_auc,
        )
    if u_auc < 0.52:
        log.warning(
            "1-U AUC=%.3f < 0.52. Uncertainty carries very little signal. "
            "Check that uncertainty_scores.pt is aligned with predictions.pt.",
            u_auc,
        )
    if cf_auc < 0.52:
        log.warning(
            "1-Delta_CF AUC=%.3f < 0.52. Counterfactual carries very little signal. "
            "Check that counterfactual_scores.pt is aligned with predictions.pt.",
            cf_auc,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_trust_calibration_v2(dataset: str) -> None:
    """Full pipeline: load real Phase 3 outputs → save calibration data."""
    log.info("=== build_trust_calibration_v2 for: %s ===", dataset)

    results_dir = Path(RESULTS_DIR) / dataset
    calib_dir   = Path(TRUST_CALIB_DIR) / dataset
    calib_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real scores ──────────────────────────────────────────────────────
    log.info("Loading uncertainty scores ...")
    u_dict = _load_uncertainty(results_dir)
    log.info("  uncertainty_scores.pt: %d queries", len(u_dict))

    log.info("Loading counterfactual scores ...")
    cf_dict = _load_counterfactual(results_dir)
    log.info("  counterfactual_scores.pt: %d queries", len(cf_dict))

    log.info("Loading provenance weights ...")
    prov_weights = _load_provenance(dataset)

    log.info("Loading predictions ...")
    pred_dict = _load_predictions(results_dir)
    if pred_dict:
        log.info("  predictions.pt: %d queries", len(pred_dict))
    else:
        # Informative error raised inside _assign_labels; raise now for clarity
        raise RuntimeError(
            "predictions.pt not found in:\n"
            f"  {results_dir}\n\n"
            "This file must be produced by Phase 3. Expected format:\n"
            "  (N, 3) LongTensor: [query_idx, true_tail_id, predicted_tail_id]\n\n"
            "To create it, add a save step to your evaluation script:\n"
            "  torch.save(predictions_tensor, results_dir / 'predictions.pt')"
        )

    log.info("Loading test triples ...")
    test_triples = _load_test_triples(dataset)
    log.info("  triples_test.pt: %d triples", len(test_triples))

    # ── Align query indices ───────────────────────────────────────────────────
    # Use the intersection of indices available in both U and Delta_CF
    common_idx = sorted(set(u_dict.keys()) & set(cf_dict.keys()))
    log.info("Common query indices (U ∩ Delta_CF): %d", len(common_idx))

    if len(common_idx) == 0:
        raise RuntimeError(
            "No common query indices between uncertainty_scores.pt and "
            "counterfactual_scores.pt. Check that both files use the same "
            "query_idx column."
        )

    # ── Load or compute filtered ranks (hetionet only) ────────────────────────
    rank_dict: Optional[Dict[int, int]] = None
    if dataset == "hetionet":
        log.info("Loading filtered test ranks ...")
        rank_dict = _load_test_ranks(results_dir)
        if rank_dict is not None:
            log.info("  test_ranks.pt: %d queries (filtered ranks)", len(rank_dict))
        else:
            log.info(
                "  test_ranks.pt not found — computing filtered ranks "
                "(re-runs model inference, ~30 min) ..."
            )
            rank_dict = _compute_filtered_ranks(dataset, results_dir)

    # ── Build aligned arrays ──────────────────────────────────────────────────
    sprov_dict = _assign_sprov(common_idx, prov_weights, test_triples, dataset, results_dir)
    label_dict = _assign_labels(common_idx, pred_dict, dataset, rank_dict=rank_dict)

    U_vals      = torch.tensor([u_dict[i]      for i in common_idx], dtype=torch.float)
    S_prov_vals = torch.tensor([sprov_dict[i]  for i in common_idx], dtype=torch.float)
    CF_vals     = torch.tensor([cf_dict[i]     for i in common_idx], dtype=torch.float)
    label_vals  = torch.tensor([label_dict[i]  for i in common_idx], dtype=torch.long)
    idx_tensor  = torch.tensor(common_idx,                           dtype=torch.long)

    # Clamp to [0, 1]
    U_vals      = U_vals.clamp(0.0, 1.0)
    S_prov_vals = S_prov_vals.clamp(0.0, 1.0)
    CF_vals     = CF_vals.clamp(0.0, 1.0)

    N = len(common_idx)
    log.info("Aligned dataset size: N=%d", N)

    # ── Associate head/relation IDs ───────────────────────────────────────────
    # test_triples[query_idx] = [head_id, rel_id, tail_id] if indices align
    n_test = test_triples.size(0)
    head_ids = torch.zeros(N, dtype=torch.long)
    rel_ids  = torch.zeros(N, dtype=torch.long)
    for i, q in enumerate(common_idx):
        if q < n_test:
            head_ids[i] = test_triples[q, 0]
            rel_ids[i]  = test_triples[q, 1]
        # else: leave as 0 (edge case where query_idx exceeds test set size)

    # ── Validate ──────────────────────────────────────────────────────────────
    log.info("Running AUC sanity checks ...")
    _validate_dataset(label_vals, U_vals, S_prov_vals, CF_vals)

    # ── Class balance ─────────────────────────────────────────────────────────
    n_pos = int(label_vals.sum().item())
    n_neg = N - n_pos
    pos_rate = n_pos / max(N, 1)
    log.info(
        "Class balance — correct=%d (%.1f%%), incorrect=%d (%.1f%%)",
        n_pos, pos_rate * 100, n_neg, (1 - pos_rate) * 100,
    )

    # ── Assemble tensors ──────────────────────────────────────────────────────
    # calibration_labels: (N, 4) [query_idx, head_id, rel_id, label]
    calibration_labels = torch.stack(
        [idx_tensor, head_ids, rel_ids, label_vals], dim=1
    )

    # calibration_components: (N, 3) [U, S_prov, Delta_CF]
    calibration_components = torch.stack([U_vals, S_prov_vals, CF_vals], dim=1)

    # ── Report class balance before saving ────────────────────────────────────
    if dataset == "hetionet":
        label_strategy = "filtered hits@10" if rank_dict is not None else "unfiltered hits@10"
    else:
        label_strategy = "top-1 accuracy"
    log.info(
        "Label strategy: %s  |  Positive rate: %.4f  "
        "N positive: %d  N total: %d",
        label_strategy,
        float(n_pos / max(N, 1)),
        n_pos,
        N,
    )
    print(
        f"  Positive rate: {n_pos / max(N, 1):.4f}  "
        f"N positive: {n_pos}  N total: {N}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    labels_path = calib_dir / "calibration_labels.pt"
    comps_path  = calib_dir / "calibration_components.pt"

    torch.save(calibration_labels,     str(labels_path))
    torch.save(calibration_components, str(comps_path))
    log.info("Saved calibration_labels.pt     shape=%s", tuple(calibration_labels.shape))
    log.info("Saved calibration_components.pt shape=%s", tuple(calibration_components.shape))

    summary = {
        "dataset": dataset,
        "num_calibration_samples": N,
        "num_positive_labels": n_pos,
        "num_negative_labels": n_neg,
        "positive_rate": float(pos_rate),
        "mean_U":        float(U_vals.mean().item()),
        "mean_S_prov":   float(S_prov_vals.mean().item()),
        "mean_Delta_CF": float(CF_vals.mean().item()),
        "is_synthetic": False,  # built from real Phase 3 model outputs
        "source": {
            "uncertainty":     str(results_dir / "uncertainty_scores.pt"),
            "counterfactual":  str(results_dir / "counterfactual_scores.pt"),
            "predictions":     str(results_dir / "predictions.pt"),
            "provenance":      str(Path(PROCESSED_DATA[dataset]) / "provenance_weights.pt"),
        },
    }
    summary_path = calib_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved calibration_summary.json (is_synthetic=False)")
    log.info("=== Done. Calibration data ready for train_trust_aggregator.py ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build trust calibration data from real Phase 3 model outputs.\n"
            "Requires: uncertainty_scores.pt, counterfactual_scores.pt, predictions.pt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fb15k237", "wn18rr", "hetionet"],
        help="Dataset to build calibration data for.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id (unused here; accepted for consistency with other scripts).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(RANDOM_SEED)
    build_trust_calibration_v2(args.dataset)


if __name__ == "__main__":
    main()
