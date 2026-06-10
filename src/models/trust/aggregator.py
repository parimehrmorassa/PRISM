"""
Attention-Based Trust Score Aggregator.

Implements the core trust formula:
    T = α(1-U) + β·S_prov + γ·Δ_CF

Where α, β, γ are learned via a query-adaptive MLP:
    [1-U, S_prov, Δ_CF] → Linear → ReLU → Dropout → Linear → ReLU → Dropout
                        → Linear → Softmax → [α, β, γ]
    T = sum([α, β, γ] * [1-U, S_prov, Δ_CF])

T ∈ [0, 1] is guaranteed by softmax weights + bounded component inputs.

Key novel contribution: α, β, γ vary by query type:
    - rare_relation  → β (provenance) highest
    - hub_entity     → γ (counterfactual) highest
    - uncertain pred → α (uncertainty) highest
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import TRUST_AGG_CONFIG

logger = logging.getLogger(__name__)


class AttentionBasedTrustAggregator(nn.Module):
    """
    MLP-based trust aggregator that learns query-adaptive attention weights α, β, γ.

    Architecture:
        Input: [1-U, S_prov, Delta_CF]  (shape: batch × 3)
        Layer 1: Linear(3 → H) → ReLU → Dropout(0.2)
        Layer 2: Linear(H → H) → ReLU → Dropout(0.2)
        Layer 3: Linear(H → 3) → Softmax  → weights [α, β, γ]
        T = dot([α, β, γ], [1-U, S_prov, Delta_CF])

    T ∈ [0, 1] is guaranteed:
        - Softmax ensures α + β + γ = 1, all ≥ 0
        - Inputs [1-U, S_prov, Delta_CF] ∈ [0, 1]
        → T = weighted sum of values in [0, 1] with non-negative weights → T ∈ [0, 1]

    Args:
        hidden_dim: Width of MLP hidden layers (from TRUST_AGG_CONFIG)
    """

    def __init__(self, hidden_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim or TRUST_AGG_CONFIG["hidden_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(TRUST_AGG_CONFIG["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(TRUST_AGG_CONFIG["dropout"]),
            nn.Linear(self.hidden_dim, 3),  # → logits for [α, β, γ]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        U: torch.Tensor,
        S_prov: torch.Tensor,
        Delta_CF: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute trust scores and attention weights.

        Args:
            U:        (batch,) — epistemic uncertainty ∈ [0, 1]
            S_prov:   (batch,) — provenance score ∈ [0, 1]
            Delta_CF: (batch,) — counterfactual sensitivity ∈ [0, 1]

        Returns:
            (T, weights):
                T:       (batch,) — trust score ∈ [0, 1]
                weights: (batch, 3) — [α, β, γ] summing to 1
        """
        # Stack inputs: [1-U, S_prov, Delta_CF] ∈ [0, 1]^3
        certainty = 1.0 - U
        x = torch.stack([certainty, S_prov, Delta_CF], dim=-1)  # (batch, 3)

        # Clip inputs to valid range
        x = x.clamp(0.0, 1.0)

        # MLP → attention weights via Softmax
        logits = self.mlp(x)                         # (batch, 3)
        weights = F.softmax(logits, dim=-1)          # (batch, 3) — α, β, γ

        # Trust score: weighted sum
        T = (weights * x).sum(dim=-1)                # (batch,)
        T = T.clamp(0.0, 1.0)

        return T, weights

    def forward_with_metadata(
        self,
        U: torch.Tensor,
        S_prov: torch.Tensor,
        Delta_CF: torch.Tensor,
        relation_freq: torch.Tensor,
        head_degree: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extended forward with query metadata for weight analysis.

        Args:
            U, S_prov, Delta_CF: Trust components (batch,)
            relation_freq: (batch,) — frequency of the query relation
            head_degree:   (batch,) — degree of the head entity

        Returns:
            (T, weights, metadata_dict)
        """
        T, weights = self.forward(U, S_prov, Delta_CF)

        # Classify query types for analysis
        rare_thresh = TRUST_AGG_CONFIG["rare_relation_freq_threshold"]
        hub_thresh = TRUST_AGG_CONFIG["hub_entity_degree_threshold"]

        query_types = []
        for i in range(len(U)):
            if relation_freq[i].item() < rare_thresh:
                query_types.append("rare_relation")
            elif head_degree[i].item() > hub_thresh:
                query_types.append("hub_entity")
            else:
                query_types.append("other")

        metadata_dict = {
            "query_types": query_types,
            "relation_freq": relation_freq.tolist(),
            "head_degree": head_degree.tolist(),
        }
        return T, weights, metadata_dict


class TrustCalibrationTrainer:
    """
    Trainer for AttentionBasedTrustAggregator.

    Trains on (U, S_prov, Delta_CF) → binary trust label pairs.
    Loss: BCE(T, label)

    Args:
        model: AttentionBasedTrustAggregator
        device: Torch device
    """

    def __init__(self, model: AttentionBasedTrustAggregator, device: torch.device,
                 pos_weight: float = 1.0):
        self.model = model
        self.device = device
        self.pos_weight = pos_weight
        # Model outputs T ∈ [0,1] (weighted average, not a logit), so we use
        # BCELoss with manual per-sample weights rather than BCEWithLogitsLoss.
        # sample_weight = pos_weight for label=1, 1 for label=0 — identical
        # semantics to BCEWithLogitsLoss(pos_weight=pos_weight).
        self._pw_tensor = torch.tensor([pos_weight])  # moved to device in _train_epoch
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRUST_AGG_CONFIG["learning_rate"],
            weight_decay=TRUST_AGG_CONFIG["weight_decay"],
        )
        self.best_auc = 0.0
        self.best_state = None

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int = None,
    ) -> List[Dict]:
        """
        Train the trust aggregator.

        Args:
            train_loader: DataLoader of (U, S_prov, Delta_CF, label)
            valid_loader: DataLoader for validation
            num_epochs: Number of training epochs

        Returns:
            history: List of epoch metric dicts
        """
        num_epochs = num_epochs or TRUST_AGG_CONFIG["num_epochs"]
        log_weight_every = 10
        patience = TRUST_AGG_CONFIG["patience"]
        patience_counter = 0
        history = []

        self.model.to(self.device)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(valid_loader)
            val_auc = val_metrics["auc_roc"]

            entry = {
                "step": epoch,
                "loss": train_loss,
                "val_auc": val_auc,
                "val_accuracy": val_metrics["accuracy"],
                "val_ece": val_metrics["ece"],
            }
            history.append(entry)

            # Log weight distributions
            if epoch % log_weight_every == 0:
                self._log_weight_distribution(valid_loader, epoch)

            logger.info(
                f"Epoch {epoch}/{num_epochs} | Loss={train_loss:.4f} | "
                f"AUC={val_auc:.4f} | Acc={val_metrics['accuracy']:.4f}"
            )

            # Save best model by validation AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info(f"Restored best model (AUC={self.best_auc:.4f})")

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            U, S_prov, Delta_CF, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            T, _ = self.model(U, S_prov, Delta_CF)
            # Weighted BCE equivalent to BCEWithLogitsLoss(pos_weight=pos_weight):
            #   label=1 → weight = pos_weight  (upweights minority class)
            #   label=0 → weight = 1
            pw = self._pw_tensor.to(self.device).item()
            sample_weights = labels.float() * (pw - 1.0) + 1.0
            loss = F.binary_cross_entropy(T, labels.float(), weight=sample_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Evaluate model on a DataLoader.

        Returns:
            dict with: accuracy, auc_roc, auc_pr, ece, f1
        """
        self.model.eval()
        all_T, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                U, S_prov, Delta_CF, labels = [b.to(self.device) for b in batch]
                T, _ = self.model(U, S_prov, Delta_CF)
                all_T.append(T.cpu())
                all_labels.append(labels.cpu())

        scores = torch.cat(all_T).numpy()
        labels = torch.cat(all_labels).numpy()

        preds = (scores >= 0.5).astype(int)
        accuracy = float(np.mean(preds == labels))

        if len(np.unique(labels)) > 1:
            auc_roc = float(roc_auc_score(labels, scores))
            auc_pr = float(average_precision_score(labels, scores))
        else:
            auc_roc = 0.5
            auc_pr = float(np.mean(labels))

        from src.evaluation.metrics import compute_expected_calibration_error
        ece = compute_expected_calibration_error(scores, labels)

        return {
            "accuracy": accuracy,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "ece": ece,
        }

    def analyze_weights_by_query_type(self, loader: DataLoader) -> Dict:
        """
        Compute mean α, β, γ grouped by query type.

        Groups:
            rare_relation: relation frequency < rare_relation_freq_threshold
            hub_entity:    head degree > hub_entity_degree_threshold
            other:         everything else

        Returns:
            dict mapping query_type → {"alpha": mean, "beta": mean, "gamma": mean}
        """
        self.model.eval()
        groups = {"rare_relation": [], "hub_entity": [], "other": []}

        rare_thresh = TRUST_AGG_CONFIG["rare_relation_freq_threshold"]
        hub_thresh = TRUST_AGG_CONFIG["hub_entity_degree_threshold"]

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    U, S_prov, Delta_CF, labels = [b.to(self.device) for b in batch]
                    rel_freq = None
                    head_deg = None
                elif len(batch) == 6:
                    U, S_prov, Delta_CF, labels, rel_freq, head_deg = [
                        b.to(self.device) for b in batch
                    ]
                else:
                    continue

                _, weights = self.model(U, S_prov, Delta_CF)  # (batch, 3)
                weights_np = weights.cpu().numpy()

                for i in range(len(U)):
                    if rel_freq is not None and head_deg is not None:
                        rf = rel_freq[i].item()
                        hd = head_deg[i].item()
                        if rf < rare_thresh:
                            qt = "rare_relation"
                        elif hd > hub_thresh:
                            qt = "hub_entity"
                        else:
                            qt = "other"
                    else:
                        qt = "other"
                    groups[qt].append(weights_np[i])

        result = {}
        for qt, weight_list in groups.items():
            if weight_list:
                arr = np.array(weight_list)  # (N, 3)
                result[qt] = {
                    "alpha": float(arr[:, 0].mean()),
                    "beta": float(arr[:, 1].mean()),
                    "gamma": float(arr[:, 2].mean()),
                    "n": len(arr),
                    "weights_array": arr,
                }
            else:
                result[qt] = {"alpha": 1/3, "beta": 1/3, "gamma": 1/3, "n": 0, "weights_array": np.array([])}

        return result

    def _log_weight_distribution(self, loader: DataLoader, epoch: int):
        """Log mean α, β, γ on validation set."""
        self.model.eval()
        all_weights = []
        with torch.no_grad():
            for batch in loader:
                U, S_prov, Delta_CF = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                _, weights = self.model(U, S_prov, Delta_CF)
                all_weights.append(weights.cpu())

        if all_weights:
            w = torch.cat(all_weights).numpy()
            logger.info(
                f"Epoch {epoch} | Mean weights: "
                f"α={w[:,0].mean():.3f} β={w[:,1].mean():.3f} γ={w[:,2].mean():.3f}"
            )
