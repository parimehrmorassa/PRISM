"""
TrustworthyKGReasoner: Complete inference API for the trustworthy KG framework.

Loads all trained components from checkpoint directory and provides
a single predict() interface that returns trust scores, reasoning paths,
and natural-language explanations.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHECKPOINT_DIR, PROCESSED_DATA, LLM_CONFIG, NBFNET_CONFIG

logger = logging.getLogger(__name__)


class TrustworthyKGReasoner:
    """
    End-to-end inference interface for the trustworthy KG reasoning framework.

    Computes:
        T = α(1-U) + β·S_prov + γ·Δ_CF

    With all components computed in a single predict() call.

    Usage:
        reasoner = TrustworthyKGReasoner.load_from_checkpoint("checkpoints/", "fb15k237")
        result = reasoner.predict("DrugA", "treats", return_explanation=True)
    """

    def __init__(
        self,
        nbfnet_model,
        trust_aggregator,
        conformal_quantifier,
        counterfactual_analyzer,
        trainer,
        explainer,
        dataset_name: str,
        num_entities: int,
        entity2id: Dict,
        id2entity: Dict,
        relation2id: Dict,
        id2relation: Dict,
    ):
        self.nbfnet = nbfnet_model
        self.aggregator = trust_aggregator
        self.quantifier = conformal_quantifier
        self.cf_analyzer = counterfactual_analyzer
        self.trainer = trainer
        self.explainer = explainer
        self.dataset_name = dataset_name
        self.num_entities = num_entities
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.relation2id = relation2id
        self.id2relation = id2relation

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: str,
        dataset_name: str,
        device: Optional[torch.device] = None,
    ) -> "TrustworthyKGReasoner":
        """
        Load all trained models from a checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint root (e.g., "checkpoints/")
            dataset_name: One of "fb15k237", "wn18rr", "hetionet"
            device: Target torch device (auto-detected if None)

        Returns:
            Fully initialized TrustworthyKGReasoner instance
        """
        from src.models.nbfnet.model import NBFNet
        from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
        from src.models.trust.aggregator import AttentionBasedTrustAggregator
        from src.models.trust.uncertainty import ConformalUncertaintyQuantifier
        from src.models.trust.counterfactual import CounterfactualAnalyzer
        from src.models.explainer.llm_interface import LLMExplainer
        from src.utils.gpu_utils import get_device

        ckpt_dir = Path(checkpoint_dir) / dataset_name
        data_dir = Path(PROCESSED_DATA[dataset_name])

        if device is None:
            device = get_device()
        logger.info(f"Loading TrustworthyKGReasoner for {dataset_name} on {device}")

        # ─── Load metadata ──────────────────────────────────────────────────────
        stats_path = data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            num_entities = stats["num_entities"]
            num_relations = stats["num_relations"]
        else:
            num_entities = 14541  # FB15k-237 default
            num_relations = 237

        entity2id, id2entity, relation2id, id2relation = {}, {}, {}, {}
        for path, d, inv in [
            (data_dir / "entity2id.json", entity2id, id2entity),
            (data_dir / "relation2id.json", relation2id, id2relation),
        ]:
            if path.exists():
                with open(path) as f:
                    loaded = json.load(f)
                d.update(loaded)
                inv.update({v: k for k, v in loaded.items()})

        # ─── NBFNet ─────────────────────────────────────────────────────────────
        nbf_path = ckpt_dir / "nbfnet_best.pt"
        if nbf_path.exists():
            nbfnet = NBFNet.load(str(nbf_path), device=device)
        else:
            logger.warning("NBFNet checkpoint not found. Using fresh model.")
            nbfnet = NBFNet(num_relations=num_relations)
        nbfnet.to(device).eval()

        # ─── Trust aggregator ───────────────────────────────────────────────────
        agg_path = ckpt_dir / "trust_aggregator.pt"
        aggregator = AttentionBasedTrustAggregator()
        if agg_path.exists():
            ckpt = torch.load(agg_path, map_location="cpu", weights_only=False)
            aggregator.load_state_dict(ckpt["state_dict"])
        aggregator.eval()

        # ─── Conformal quantifier ───────────────────────────────────────────────
        q_path = ckpt_dir / "conformal_quantile.pkl"
        if q_path.exists():
            quantifier = ConformalUncertaintyQuantifier.load(str(q_path))
            quantifier.num_entities = num_entities
        else:
            quantifier = ConformalUncertaintyQuantifier(num_entities=num_entities)
            quantifier.quantile = 0.8  # default fallback

        # ─── Trainer (for subgraph extraction) ──────────────────────────────────
        prov_path = data_dir / "provenance_weights.pt"
        triples_path = data_dir / "triples_train.pt"

        if triples_path.exists():
            all_triples = torch.load(triples_path, weights_only=True).long()
            all_edge_index = torch.stack([all_triples[:, 0], all_triples[:, 2]], dim=0)
            all_edge_type = all_triples[:, 1]
        else:
            all_edge_index = torch.zeros(2, 0, dtype=torch.long)
            all_edge_type = torch.zeros(0, dtype=torch.long)
            all_triples = torch.zeros(0, 3, dtype=torch.long)

        if prov_path.exists():
            all_edge_prov = torch.load(prov_path, weights_only=True).float()
        else:
            all_edge_prov = torch.ones(all_edge_index.shape[1])

        all_triples_set = set()
        for row in all_triples:
            all_triples_set.add((row[0].item(), row[1].item(), row[2].item()))

        trainer = FullGraphNBFNetTrainer(
            model=nbfnet,
            device=device,
            dataset_name=dataset_name,
            all_triples=all_triples_set,
            num_entities=num_entities,
            all_edge_index=all_edge_index,
            all_edge_type=all_edge_type,
            all_edge_prov=all_edge_prov,
        )

        # ─── Analyzers and explainer ────────────────────────────────────────────
        cf_analyzer = CounterfactualAnalyzer(model=nbfnet, device=device)
        explainer = LLMExplainer(config=LLM_CONFIG)

        return cls(
            nbfnet_model=nbfnet,
            trust_aggregator=aggregator,
            conformal_quantifier=quantifier,
            counterfactual_analyzer=cf_analyzer,
            trainer=trainer,
            explainer=explainer,
            dataset_name=dataset_name,
            num_entities=num_entities,
            entity2id=entity2id,
            id2entity=id2entity,
            relation2id=relation2id,
            id2relation=id2relation,
        )

    def predict(
        self,
        head: str,
        relation: str,
        tail: Optional[str] = None,
        return_explanation: bool = True,
        domain: str = "general",
    ) -> Dict:
        """
        Run full trustworthy prediction for a KG query.

        Args:
            head: Head entity name (must be in entity2id)
            relation: Relation name (must be in relation2id)
            tail: Optional true tail (for evaluation). If None, uses top-scoring entity.
            return_explanation: Whether to generate LLM explanation
            domain: Domain context for explanation generation

        Returns:
            {
                "predicted_tail": str,
                "confidence": float,
                "trust_score": float,
                "trust_breakdown": {
                    "uncertainty": float,
                    "provenance": float,
                    "counterfactual": float,
                    "weights": {"alpha": float, "beta": float, "gamma": float}
                },
                "reasoning_paths": List[dict],
                "explanation": str  (if return_explanation=True)
            }
        """
        from src.models.trust.provenance_aggregator import compute_path_provenance_from_attention
        from src.utils.gpu_utils import autocast_ctx, get_amp_dtype

        device = next(self.nbfnet.parameters()).device
        use_amp = NBFNET_CONFIG["use_amp"] and device.type == "cuda"
        amp_dtype = get_amp_dtype(NBFNET_CONFIG["amp_dtype"])

        # Map string names to IDs
        head_id = self.entity2id.get(head)
        relation_id = self.relation2id.get(relation)

        if head_id is None:
            logger.warning(f"Unknown entity: '{head}'. Using ID=0.")
            head_id = 0
        if relation_id is None:
            logger.warning(f"Unknown relation: '{relation}'. Using ID=0.")
            relation_id = 0

        tail_id = self.entity2id.get(tail, 0) if tail else 0

        # Extract subgraph
        subgraph = self.trainer._extract_subgraph(head_id, relation_id, tail_id)
        if subgraph is None:
            return self._empty_result(head, relation, tail or "unknown")

        edge_index = subgraph["edge_index"].to(device)
        edge_type = subgraph["edge_type"].to(device)
        edge_prov = subgraph["edge_prov"].to(device)
        num_nodes = subgraph["num_nodes"]

        # Run NBFNet
        self.nbfnet.eval()
        with torch.no_grad():
            with autocast_ctx(device, use_amp, amp_dtype):
                scores, attn_weights = self.nbfnet(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_prov=edge_prov,
                    query_head=subgraph["local_head"],
                    query_relation=relation_id,
                    num_nodes=num_nodes,
                )

        # Top-1 prediction
        top_local = scores.argmax().item()
        l2g = {v: k for k, v in subgraph["global_to_local"].items()}
        top_global = l2g.get(top_local, top_local)
        predicted_tail_name = self.id2entity.get(top_global, str(top_global))
        confidence = scores[top_local].item()

        # If tail specified, use it
        actual_tail_id = self.entity2id.get(tail, top_global) if tail else top_global
        actual_local = subgraph["global_to_local"].get(actual_tail_id, top_local)
        tail_score = scores[actual_local].item()

        # ─── Trust Components ────────────────────────────────────────────────────
        # Uncertainty
        U, pred_set = self.quantifier.compute_uncertainty_from_scores(
            scores, num_entities=self.num_entities
        )

        # Provenance (from top attention edges)
        S_prov, path_prov_scores = compute_path_provenance_from_attention(
            edge_index, edge_prov, attn_weights, top_k=5
        )

        # Counterfactual
        critical_edges, baseline_score = self.cf_analyzer.identify_critical_edges(
            subgraph, relation_id
        )
        Delta_CF, edge_impacts = self.cf_analyzer.compute_sensitivity(
            subgraph, relation_id, critical_edges, baseline_score
        )

        # Trust score aggregation
        U_t = torch.tensor([U])
        S_t = torch.tensor([S_prov])
        CF_t = torch.tensor([Delta_CF])
        with torch.no_grad():
            T, weights = self.aggregator(U_t, S_t, CF_t)
        T = T.item()
        w = weights[0].tolist()

        # Reasoning paths
        reasoning_paths = self.nbfnet.extract_reasoning_paths(
            edge_index=edge_index,
            edge_type=edge_type,
            edge_prov=edge_prov,
            all_attention_weights=attn_weights,
            query_head=subgraph["local_head"],
            query_tail=actual_local,
            id2entity=self.id2entity,
            id2relation=self.id2relation,
            top_k=5,
        )

        # Critical edge name
        most_critical = critical_edges[0] if critical_edges else {}
        max_delta = max(edge_impacts.get("critical", [0.0]) or [0.0])

        result = {
            "predicted_tail": predicted_tail_name,
            "confidence": float(confidence),
            "trust_score": float(T),
            "trust_breakdown": {
                "uncertainty": float(U),
                "provenance": float(S_prov),
                "counterfactual": float(Delta_CF),
                "weights": {
                    "alpha": float(w[0]),
                    "beta": float(w[1]),
                    "gamma": float(w[2]),
                },
            },
            "reasoning_paths": reasoning_paths,
        }

        # ─── Explanation ─────────────────────────────────────────────────────────
        if return_explanation:
            query_result = {
                "query": (head, relation, predicted_tail_name),
                "head": head,
                "relation": relation,
                "prediction": predicted_tail_name,
                "confidence": confidence,
                "trust_score": T,
                "trust_breakdown": result["trust_breakdown"],
                "reasoning_paths": reasoning_paths,
                "critical_edge": most_critical,
                "max_delta": max_delta,
            }
            explanation = self.explainer.generate_explanation(query_result, domain=domain)
            result["explanation"] = explanation

        return result

    def _empty_result(self, head: str, relation: str, tail: str) -> Dict:
        """Return a safe default result when subgraph extraction fails."""
        return {
            "predicted_tail": tail,
            "confidence": 0.0,
            "trust_score": 0.0,
            "trust_breakdown": {
                "uncertainty": 1.0,
                "provenance": 0.0,
                "counterfactual": 0.0,
                "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
            },
            "reasoning_paths": [],
            "explanation": "Unable to compute prediction due to missing subgraph.",
        }
