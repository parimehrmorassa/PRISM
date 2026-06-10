"""
Generate LLM explanations for test queries with LExT scoring.

Usage:
    python scripts/generate_explanations.py --dataset fb15k237 --num_samples 200
    python scripts/generate_explanations.py --dataset hetionet --domain medical
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, LLM_CONFIG, EVALUATION_CONFIG
)
from src.data.dataset import build_dataloaders
from src.models.nbfnet.model import NBFNet
from src.models.nbfnet.trainer import FullGraphNBFNetTrainer
from src.models.trust.aggregator import AttentionBasedTrustAggregator
from src.models.trust.uncertainty import ConformalUncertaintyQuantifier
from src.models.trust.counterfactual import CounterfactualAnalyzer
from src.models.trust.provenance_aggregator import compute_path_provenance_from_attention
from src.models.explainer.llm_interface import LLMExplainer
from src.utils.gpu_utils import get_device
from src.utils.logging_utils import setup_logging, format_metrics_table

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate explanations for KG predictions")
    parser.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr", "hetionet"])
    parser.add_argument("--num_samples", type=int, default=EVALUATION_CONFIG["num_explanation_samples"])
    parser.add_argument("--domain", type=str, default="general",
                        choices=["general", "medical", "biology", "social"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_query_result(
    head_id, relation_id, tail_id, query_idx,
    model, trainer, aggregator, quantifier, analyzer,
    train_ds, device, num_entities
):
    """Build a full query_result dict for explanation generation."""
    h, r, t = head_id, relation_id, tail_id

    subgraph = trainer._extract_subgraph(h, r, t)
    if subgraph is None:
        return None

    edge_index = subgraph["edge_index"].to(device)
    edge_type = subgraph["edge_type"].to(device)
    edge_prov = subgraph["edge_prov"].to(device)
    num_nodes = subgraph["num_nodes"]

    with torch.no_grad():
        scores, attn_weights = model(
            edge_index=edge_index,
            edge_type=edge_type,
            edge_prov=edge_prov,
            query_head=subgraph["local_head"],
            query_relation=r,
            num_nodes=num_nodes,
        )

    # Trust components
    confidence = scores[subgraph["local_tail"]].item()
    U, pred_set = quantifier.compute_uncertainty_from_scores(scores, num_entities=num_entities)

    S_prov, path_prov_scores = compute_path_provenance_from_attention(
        edge_index, edge_prov, attn_weights
    )

    critical_edges, baseline_score = analyzer.identify_critical_edges(subgraph, r)
    Delta_CF, edge_impacts = analyzer.compute_sensitivity(
        subgraph, r, critical_edges, baseline_score
    )

    # Aggregate trust score
    U_t = torch.tensor([U])
    S_t = torch.tensor([S_prov])
    CF_t = torch.tensor([Delta_CF])
    with torch.no_grad():
        T, weights = aggregator(U_t, S_t, CF_t)
    T = T.item()
    w = weights[0].tolist()

    # Reasoning paths
    id2entity = train_ds.id2entity
    id2relation = train_ds.id2relation
    reasoning_paths = model.extract_reasoning_paths(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_prov=edge_prov,
        all_attention_weights=attn_weights,
        query_head=subgraph["local_head"],
        query_tail=subgraph["local_tail"],
        id2entity=id2entity,
        id2relation=id2relation,
        top_k=5,
    )

    # Critical edge info
    most_critical = critical_edges[0] if critical_edges else {}
    max_delta = max(edge_impacts.get("critical", [0.0]) or [0.0])

    head_name = id2entity.get(h, str(h))
    relation_name = id2relation.get(r, str(r))
    tail_name = id2entity.get(t, str(t))

    critical_edge_str = "N/A"
    if most_critical:
        ch = id2entity.get(most_critical.get("head", -1), str(most_critical.get("head", "?")))
        cr = id2relation.get(most_critical.get("relation", -1), str(most_critical.get("relation", "?")))
        ct = id2entity.get(most_critical.get("tail", -1), str(most_critical.get("tail", "?")))
        critical_edge_str = f"({ch}) --[{cr}]--> ({ct})"

    return {
        "query_idx": query_idx,
        "query": (head_name, relation_name, tail_name),
        "head": head_name,
        "relation": relation_name,
        "prediction": tail_name,
        "confidence": confidence,
        "trust_score": T,
        "trust_breakdown": {
            "uncertainty": U,
            "provenance": S_prov,
            "counterfactual": Delta_CF,
            "weights": {"alpha": w[0], "beta": w[1], "gamma": w[2]},
        },
        "reasoning_paths": reasoning_paths,
        "critical_edge": critical_edge_str,
        "max_delta": max_delta,
        "metadata": {
            "relation_id": r,
            "head_id": h,
            "tail_id": t,
        },
    }


def main():
    args = parse_args()
    setup_logging()

    device = get_device(args.gpu)
    results_dir = Path(RESULTS_DIR) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load dataset ───────────────────────────────────────────────────────────
    _, valid_loader, test_loader, train_ds = build_dataloaders(
        args.dataset, batch_size=64, num_workers=0, device="cpu"
    )
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    # ─── Load models ────────────────────────────────────────────────────────────
    ckpt_dir = Path(CHECKPOINT_DIR) / args.dataset
    nbfnet_path = ckpt_dir / "nbfnet_best.pt"
    if nbfnet_path.exists():
        model = NBFNet.load(str(nbfnet_path), device=device)
    else:
        logger.warning("NBFNet checkpoint not found. Using fresh model.")
        model = NBFNet(num_relations=num_relations)
    model.to(device).eval()

    # Trust aggregator
    agg_path = ckpt_dir / "trust_aggregator.pt"
    aggregator = AttentionBasedTrustAggregator()
    if agg_path.exists():
        ckpt = torch.load(agg_path, map_location="cpu", weights_only=False)
        aggregator.load_state_dict(ckpt["state_dict"])
    aggregator.eval()

    # Conformal quantifier
    q_path = ckpt_dir / "conformal_quantile.pkl"
    if q_path.exists():
        quantifier = ConformalUncertaintyQuantifier.load(str(q_path))
        quantifier.num_entities = num_entities
    else:
        quantifier = ConformalUncertaintyQuantifier(num_entities=num_entities)
        quantifier.quantile = 0.8  # default fallback

    # Trainer and analyzer
    all_triples = train_ds.get_all_triples_tensor()
    all_edge_index = torch.stack([all_triples[:, 0], all_triples[:, 2]], dim=0)
    trainer = FullGraphNBFNetTrainer(
        model=model, device=device, dataset_name=args.dataset,
        all_triples=train_ds.true_triples, num_entities=num_entities,
        all_edge_index=all_edge_index,
        all_edge_type=all_triples[:, 1],
        all_edge_prov=train_ds.get_all_provenance(),
    )
    analyzer = CounterfactualAnalyzer(model=model, device=device)

    # LLM explainer
    explainer = LLMExplainer(config=LLM_CONFIG)

    # ─── Generate explanations ───────────────────────────────────────────────────
    all_explanations = []
    num_faithful = 0
    num_samples = args.num_samples
    processed = 0

    logger.info(f"Generating explanations for {num_samples} samples...")
    for batch in tqdm(test_loader, desc="Generating"):
        for i in range(len(batch["head"])):
            if processed >= num_samples:
                break

            h = batch["head"][i].item()
            r = batch["relation"][i].item()
            t = batch["tail"][i].item()
            q_idx = batch["query_idx"][i].item()

            qr = build_query_result(
                h, r, t, q_idx, model, trainer, aggregator, quantifier,
                analyzer, train_ds, device, num_entities
            )
            if qr is None:
                continue

            explanation = explainer.generate_explanation(qr, domain=args.domain)
            is_faithful, reason = explainer.verify_faithfulness(explanation, qr)

            if is_faithful:
                num_faithful += 1

            qr["explanation"] = explanation
            qr["is_faithful"] = is_faithful
            qr["faithfulness_reason"] = reason

            # Convert tensors to serializable
            for path in qr.get("reasoning_paths", []):
                for k, v in list(path.items()):
                    if hasattr(v, "item"):
                        path[k] = v.item()

            all_explanations.append(qr)
            processed += 1

        if processed >= num_samples:
            break

    # ─── Save results ────────────────────────────────────────────────────────────
    out_path = results_dir / "explanations.json"
    with open(out_path, "w") as f:
        json.dump(all_explanations, f, indent=2, default=str)
    logger.info(f"Saved {len(all_explanations)} explanations to {out_path}")

    # ─── Print statistics ────────────────────────────────────────────────────────
    import numpy as np
    pct_faithful = 100.0 * num_faithful / max(len(all_explanations), 1)
    word_counts = [len(e.get("explanation", "").split()) for e in all_explanations]
    mean_words = float(np.mean(word_counts)) if word_counts else 0.0

    stats = {
        "num_generated": len(all_explanations),
        "pct_faithful": pct_faithful,
        "mean_word_count": mean_words,
    }
    print("\n" + format_metrics_table(stats, title=f"Explanation Statistics — {args.dataset}"))

    with open(results_dir / "explanation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Faithful: {pct_faithful:.1f}%")


if __name__ == "__main__":
    main()
