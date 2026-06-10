# A Multi-Signal Trust Framework for Selective Knowledge Graph Reasoning in Service Ecosystems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** *Counterfactual and Provenance-Aware Trustworthy Framework for Knowledge Graph Reasoning*
> Submitted to ICWS 2026

---

## Overview

**EnTrust-WS** is a trust-aware selective prediction framework for knowledge graph link prediction. It augments a Neural Bellman-Ford Network (NBFNet) backbone with three complementary trust signals and a learned query-adaptive aggregator, enabling a service operator to set a precision threshold without retraining.

### Key Results

| Dataset | Threshold τ | Coverage | Hits@1 | Improvement |
|---------|------------|----------|--------|-------------|
| FB15k-237 | 0.9 | 0.4% | 0.835 | +244% |
| Hetionet  | 0.7 | 0.1% | 0.861 | +745% |
| Hetionet  | 0.9 | <0.1% | 0.943 | +825% |

- **ECE = 0.034** after temperature scaling (FB15k-237)
- **100% explanation faithfulness** across 200 sampled predictions
- Adaptive weights outperform fixed uniform weights by **+0.110 AUC**

---

## Framework Architecture

```
Input: Query (h, r, ?)
         │
         ▼
┌─────────────────────────────┐
│  Provenance-Weighted NBFNet  │  ← edge weights from source metadata
│  6-layer message passing     │
│  hidden_dim = 64             │
└──────────────┬──────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
┌────────┐ ┌──────┐ ┌──────┐
│  U     │ │Sprov │ │ ΔCF  │
│Margin  │ │Path  │ │Edge  │
│Uncert. │ │Prov. │ │Ablat.│
└────┬───┘ └──┬───┘ └──┬───┘
     └────────┼────────┘
              ▼
   ┌─────────────────────┐
   │  Attention-Based    │
   │  Trust Aggregator   │  ← learned α, β, γ per query
   │  T = α(1-U) +       │
   │      β·Sprov +      │
   │      γ·ΔCF          │
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  Selective Predictor │  ← abstain if T < τ
   │  + LLM Explanation  │
   └─────────────────────┘
```

### Three Trust Signals

| Signal | Formula | Captures |
|--------|---------|----------|
| Epistemic Uncertainty | $U = s_2 / s_1$ | Model confidence (score margin) |
| Provenance Quality | $S_\text{prov} = \sum \hat{a}_i \cdot \min_{e \in \pi_i} w_e$ | Source reliability along reasoning paths |
| Counterfactual Robustness | $\Delta_{CF} = (f - f_{\setminus e^*}) / f$ | Sensitivity to critical edge removal |

---

## Project Structure

```
trustworthy-kg-reasoning/
├── data/
│   ├── raw/                        # Original dataset files
│   ├── processed/
│   │   ├── fb15k237/               # Processed triples + provenance
│   │   └── hetionet/
│   └── trust_calibration/          # Trust calibration datasets
│       ├── fb15k237/
│       └── hetionet/
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataset loading
│   │   ├── provenance.py           # Provenance weight generation
│   │   └── augmentation.py        # Trust calibration dataset builder
│   │
│   ├── models/
│   │   ├── nbfnet/
│   │   │   ├── model.py            # NBFNet architecture
│   │   │   ├── layer.py            # Provenance-aware message passing
│   │   │   └── trainer.py          # Training loop
│   │   │
│   │   └── trust/
│   │       ├── uncertainty.py      # Score-margin uncertainty
│   │       ├── counterfactual.py   # Structural edge ablation
│   │       └── aggregator.py      # Attention-based trust aggregator
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Link prediction + trust metrics
│   │   └── calibration.py         # ECE, reliability diagrams
│   │
│   └── config.py                   # Hyperparameter configuration
│
├── scripts/
│   ├── train_nbfnet.py             # Phase 2: NBFNet training
│   ├── compute_uncertainty.py      # Phase 3A: Uncertainty scores
│   ├── compute_counterfactuals.py  # Phase 3B: ΔCF scores
│   ├── train_trust_aggregator.py   # Phase 4: Trust aggregator
│   ├── evaluate_system.py          # Phase 6: Full evaluation
│   └── ablation_train.py          # Phase 7: Ablation studies
│
├── experiments/
│   ├── checkpoints/                # Model checkpoints
│   └── results/                    # Evaluation outputs
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/entrust-ws.git
cd entrust-ws

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
tqdm>=4.65.0
```

---

## Datasets

### FB15k-237

```bash
python3 scripts/download_fb15k237.py
python3 scripts/generate_provenance.py --dataset fb15k237
```

### Hetionet

Download from [het.io](https://het.io/):

```bash
python3 scripts/generate_provenance.py --dataset hetionet
```

Hetionet provenance scores are derived directly from source-database
confidence fields (DrugBank, PubMed, UniProt).

---

## Training

### Step 1 — Train NBFNet

```bash
python3 scripts/train_nbfnet.py \
  --dataset fb15k237 \
  --hidden_dim 64 \
  --num_layers 6 \
  --lr 1e-4 \
  --batch_size 512 \
  --patience 15
```

Expected results:

| Dataset | MRR | H@1 | H@10 |
|---------|-----|-----|------|
| FB15k-237 | 0.302 | 0.243 | 0.401 |
| Hetionet  | 0.216 | 0.102 | 0.408 |

### Step 2 — Compute Trust Signals

```bash
# Uncertainty scores
python3 scripts/compute_uncertainty.py --dataset fb15k237

# Counterfactual sensitivity
python3 scripts/compute_counterfactuals.py --dataset fb15k237 --top_k 10
```

### Step 3 — Train Trust Aggregator

```bash
python3 scripts/train_trust_aggregator.py \
  --dataset fb15k237 \
  --hidden_dim 64 \
  --num_epochs 50 \
  --lr 1e-3
```

Expected results after temperature scaling:

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.682 |
| ECE | 0.034 |
| Temperature θ | 1.38 |
| α (uncertainty) | 0.512 |
| β (provenance) | 0.396 |
| γ (counterfactual) | 0.091 |

### Step 4 — Full Evaluation

```bash
python3 scripts/evaluate_system.py \
  --dataset fb15k237 \
  --trust_model checkpoints/fb15k237/trust_aggregator.pt
```

---

## Selective Prediction

The trust score enables threshold-based deployment without retraining.
For a target precision $P^*$, select:

$$\tau^* = \inf\{\tau : \Pr[\hat{t}=t \mid T_\text{cal} \geq \tau] \geq P^*\}$$

### FB15k-237 Results (baseline H@1 = 0.2429)

| τ | Coverage | Learned H@1 | Fixed H@1 |
|---|----------|-------------|-----------|
| 0.3 | 77.9% / 35.2% | 0.2696 (+11.0%) | 0.2050 (−15.6%) |
| 0.5 | 17.3% / 10.9% | 0.2740 (+12.8%) | 0.1964 (−19.1%) |
| 0.7 |  5.0% /  1.0% | 0.3170 (+30.5%) | 0.4467 (+83.9%) |
| 0.9 |  0.4% /  0.2% | **0.8352 (+243.8%)** | 0.7561 (+211.3%) |

### Hetionet Results (baseline H@1 = 0.1019)

| τ | Coverage | Learned H@1 | Fixed H@1 |
|---|----------|-------------|-----------|
| 0.5 | 0.2% / 2.6% | 0.3352 (+229.0%) | 0.2254 (+121.2%) |
| 0.7 | 0.1% / 0.1% | 0.8607 (+744.5%) | 0.6319 (+520.0%) |
| 0.9 | <0.1% / <0.1% | **0.9429 (+825.2%)** | 0.9385 (+820.9%) |

All improvements at τ=0.9 confirmed by 95% Wilson CI: [0.746, 0.898], n=91.

---

## Ablation Study

| Configuration | AUC-ROC | ECE | Sel. H@1 (τ=0.9) |
|--------------|---------|-----|------------------|
| Full model | **0.682** | **0.034** | 0.835 |
| w/o uncertainty | 0.597 | 0.390 | 0.506 |
| w/o provenance | 0.667 | 0.107 | 0.854† |
| w/o counterfactual | 0.638 | 0.338 | 0.790 |
| Fixed weights (1/3) | 0.572 | 0.178 | 0.756 |

†No-provenance Sel. H@1 exceeds full model at τ=0.9 due to different
accepted query subsets (89 vs. 91); full model dominates in AUPC
(0.284 vs. 0.238, −16.3%).

Run ablations:

```bash
python3 scripts/ablation_train.py \
  --dataset fb15k237 \
  --zero_component uncertainty   # or provenance / counterfactual
```

---

## Explanation Generation

Faithful natural-language explanations are generated via structured
LLM prompting. Every entity mention is verified against the model's
own reasoning paths.

```python
from src.models.trust.aggregator import AttentionBasedTrustAggregator
from src.explainer.llm_interface import LLMExplainer

explainer = LLMExplainer(api_key="your-key")
explanation = explainer.generate_explanation(
    query={"head": "Aspirin", "relation": "treats"},
    trust_components={"U": 0.12, "S_prov": 0.84, "Delta_CF": 0.03},
    reasoning_paths=paths,
    trust_score=0.91
)
```

**100% faithfulness** across 200 sampled FB15k-237 predictions
(Definition 1: all entity mentions in explanation ⊆ reasoning path nodes).

---

## GPU Requirements

| Task | Memory | Time |
|------|--------|------|
| NBFNet training (FB15k-237) | ~8 GB | 4–8 hours |
| NBFNet training (Hetionet) | ~16 GB | 8–12 hours |
| Counterfactual analysis | ~8 GB | 2–4 hours |
| Trust aggregator training | ~2 GB | 30 minutes |

All experiments in the paper were run on a single **H200 NVL GPU (143 GB VRAM)**.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{entrustwsicws2026,
  title     = {Counterfactual and Provenance-Aware Trustworthy
               Framework for Knowledge Graph Reasoning},
  author    = {[Author]},
  booktitle = {Proceedings of the IEEE International Conference
               on Web Services (ICWS)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- NBFNet implementation based on
  [DeepGraphLearning/NBFNet](https://github.com/DeepGraphLearning/NBFNet)
- Hetionet data from [het.io](https://het.io/)
- FB15k-237 from
  [Toutanova & Chen, 2015](https://aclanthology.org/W15-4007/)
