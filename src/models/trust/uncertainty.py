"""
Conformal Uncertainty Quantifier for Knowledge Graph link prediction.

Uses split conformal prediction to provide:
  - Calibrated prediction sets with ~(1-alpha) coverage guarantee
  - Normalized uncertainty U = |prediction_set| / num_entities ∈ [0, 1]

Formula:
    nonconformity score s_i = 1 - score(h, r, t_true)
    quantile q = ceil((n+1)(1-alpha)/n)-th quantile of calibration scores
    prediction set = {e | score(h,r,e) >= 1 - q}
    U = |prediction_set| / num_entities
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import UNCERTAINTY_CONFIG
from src.utils.gpu_utils import autocast_ctx, get_amp_dtype
from config import NBFNET_CONFIG

logger = logging.getLogger(__name__)


class ConformalUncertaintyQuantifier:
    """
    Split conformal prediction for knowledge graph link prediction.

    After calibration on a held-out set, provides:
      - Uncertainty U ∈ [0, 1] for any query
      - Prediction set with marginal coverage guarantee ≥ 1 - alpha

    Args:
        alpha: Target miscoverage rate. Default 0.1 → ~90% coverage.
        num_entities: Total number of entities (for normalization).
    """

    def __init__(self, alpha: float = None, num_entities: int = None):
        self.alpha = alpha if alpha is not None else UNCERTAINTY_CONFIG["alpha"]
        self.num_entities = num_entities
        self.quantile: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None

    def calibrate(
        self,
        model,
        calibration_loader: DataLoader,
        device: torch.device,
        trainer,
        num_entities: int = None,
    ):
        """
        Calibrate the conformal quantile from a calibration (validation) set.

        For each (h, r, t) in calibration:
            s_i = 1 - score(h, r, t_true)

        Then:
            q = np.quantile(scores, ceil((n+1)(1-alpha)/n) / n)

        Args:
            model: Trained NBFNet model
            calibration_loader: DataLoader for calibration triples
            device: Torch device
            trainer: NBFNetTrainer (for subgraph extraction)
            num_entities: Total entity count (for normalization)
        """
        if num_entities is not None:
            self.num_entities = num_entities

        model.eval()
        use_amp = NBFNET_CONFIG["use_amp"] and device.type == "cuda"
        amp_dtype = get_amp_dtype(NBFNET_CONFIG["amp_dtype"])

        nonconformity_scores = []

        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc="Calibrating conformal"):
                heads = batch["head"]
                rels = batch["relation"]
                tails = batch["tail"]

                for i in range(len(heads)):
                    h, r, t = heads[i].item(), rels[i].item(), tails[i].item()

                    subgraph = trainer._extract_subgraph(h, r, t)
                    if subgraph is None:
                        nonconformity_scores.append(1.0)
                        continue

                    edge_index = subgraph["edge_index"].to(device)
                    edge_type = subgraph["edge_type"].to(device)
                    edge_prov = subgraph["edge_prov"].to(device)
                    local_head = subgraph["local_head"]
                    local_tail = subgraph["local_tail"]
                    num_nodes = subgraph["num_nodes"]

                    with autocast_ctx(device, use_amp, amp_dtype):
                        scores, _ = model(
                            edge_index=edge_index,
                            edge_type=edge_type,
                            edge_prov=edge_prov,
                            query_head=local_head,
                            query_relation=r,
                            num_nodes=num_nodes,
                        )

                    true_score = scores[local_tail].item()
                    # Nonconformity: how "surprising" is the true answer?
                    s_i = 1.0 - true_score
                    nonconformity_scores.append(s_i)

        scores_arr = np.array(nonconformity_scores)
        self._calibration_scores = scores_arr

        n = len(scores_arr)
        # Standard conformal quantile (corrected finite-sample)
        level = min((n + 1) * (1 - self.alpha) / n, 1.0)
        self.quantile = float(np.quantile(scores_arr, level))

        logger.info(
            f"Conformal calibration complete: n={n}, alpha={self.alpha}, "
            f"quantile={self.quantile:.4f}"
        )

    def compute_uncertainty(
        self,
        model,
        query_head: int,
        query_relation: int,
        trainer,
        device: torch.device,
        all_candidate_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[float, List[int]]:
        """
        Compute uncertainty U and prediction set for a single query.

        U = |prediction_set| / num_entities ∈ [0, 1]

        Args:
            model: Trained NBFNet
            query_head: Head entity index
            query_relation: Relation index
            trainer: NBFNetTrainer for subgraph extraction
            device: Torch device
            all_candidate_scores: Pre-computed scores if available (N,)

        Returns:
            (U, prediction_set): U ∈ [0, 1], prediction_set as list of entity IDs
        """
        if self.quantile is None:
            raise RuntimeError("Call calibrate() before compute_uncertainty()")

        if all_candidate_scores is not None:
            scores = all_candidate_scores
        else:
            # Run model on the query subgraph
            model.eval()
            use_amp = NBFNET_CONFIG["use_amp"] and device.type == "cuda"
            amp_dtype = get_amp_dtype(NBFNET_CONFIG["amp_dtype"])

            # Use a dummy tail for subgraph extraction
            subgraph = trainer._extract_subgraph(query_head, query_relation, 0)
            if subgraph is None:
                return 1.0, list(range(min(10, self.num_entities or 10)))

            edge_index = subgraph["edge_index"].to(device)
            edge_type = subgraph["edge_type"].to(device)
            edge_prov = subgraph["edge_prov"].to(device)
            num_nodes = subgraph["num_nodes"]

            with torch.no_grad():
                with autocast_ctx(device, use_amp, amp_dtype):
                    scores, _ = model(
                        edge_index=edge_index,
                        edge_type=edge_type,
                        edge_prov=edge_prov,
                        query_head=subgraph["local_head"],
                        query_relation=query_relation,
                        num_nodes=num_nodes,
                    )

        # Threshold: include e if score(h,r,e) >= 1 - quantile
        threshold = 1.0 - self.quantile
        g2l = subgraph.get("global_to_local", {}) if all_candidate_scores is None else {}
        l2g = {v: k for k, v in g2l.items()} if g2l else {}

        prediction_set = []
        for local_idx, score_val in enumerate(scores):
            if score_val.item() >= threshold:
                global_idx = l2g.get(local_idx, local_idx)
                prediction_set.append(global_idx)

        num_ents = self.num_entities or max(len(scores), 1)
        U = len(prediction_set) / num_ents
        U = max(0.0, min(1.0, U))  # clip to [0, 1]

        assert 0.0 <= U <= 1.0, f"U out of range: {U}"
        return U, prediction_set

    def compute_uncertainty_from_scores(
        self,
        scores: torch.Tensor,
        num_entities: int = None,
    ) -> Tuple[float, List[int]]:
        """
        Compute U directly from a pre-computed score tensor.

        Args:
            scores: (num_nodes,) tensor of scores ∈ [0, 1]
            num_entities: Total entity count for normalization

        Returns:
            (U, prediction_set_indices)
        """
        if self.quantile is None:
            raise RuntimeError("Call calibrate() before computing uncertainty.")

        threshold = 1.0 - self.quantile
        pred_mask = scores >= threshold
        prediction_set = pred_mask.nonzero(as_tuple=True)[0].tolist()

        num_ents = num_entities or self.num_entities or max(len(scores), 1)
        U = len(prediction_set) / num_ents
        U = max(0.0, min(1.0, U))

        assert 0.0 <= U <= 1.0, f"U out of range: {U}"
        return U, prediction_set

    def save(self, path: str):
        """Serialize the calibrated quantile to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "quantile": self.quantile,
                    "alpha": self.alpha,
                    "num_entities": self.num_entities,
                    "calibration_scores": self._calibration_scores,
                },
                f,
            )
        logger.info(f"Conformal quantifier saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalUncertaintyQuantifier":
        """Load from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(alpha=data["alpha"], num_entities=data["num_entities"])
        obj.quantile = data["quantile"]
        obj._calibration_scores = data.get("calibration_scores")
        logger.info(f"Conformal quantifier loaded from {path} (quantile={obj.quantile:.4f})")
        return obj
