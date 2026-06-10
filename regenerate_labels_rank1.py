"""
regenerate_labels_rank1.py

Rebuild calibration labels using rank == 1 as the positive condition,
consistent with the paper's claim that the trust score estimates P(rank 1).

Reads:
  experiments/results/fb15k237/test_ranks.pt    [N, 2] (query_idx, filtered_rank)
  trust_calibration/fb15k237/calibration_labels.pt  [N, 4] (idx, head, rel, old_label)

Writes:
  trust_calibration/fb15k237/calibration_labels_rank1.pt  [N, 4] same structure, new col 3

Does NOT overwrite the original calibration_labels.pt.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

RANKS_PATH  = Path("experiments/results/fb15k237/test_ranks.pt")
LABELS_PATH = Path("trust_calibration/fb15k237/calibration_labels.pt")
OUT_PATH    = Path("trust_calibration/fb15k237/calibration_labels_rank1.pt")


def main():
    ranks  = torch.load(RANKS_PATH,  weights_only=True)   # [N, 2]: [query_idx, filtered_rank]
    labels = torch.load(LABELS_PATH, weights_only=True)   # [N, 4]: [idx, head, rel, old_label]

    N = len(ranks)
    assert len(labels) == N, f"Size mismatch: ranks={N}, labels={len(labels)}"

    # Build query_idx → filtered_rank lookup
    rank_dict = {int(ranks[i, 0].item()): int(ranks[i, 1].item()) for i in range(N)}

    # Rebuild label col: 1 if filtered_rank == 1, else 0
    new_labels = labels.clone()
    for i in range(N):
        qidx = int(labels[i, 0].item())
        filtered_rank = rank_dict.get(qidx, -1)
        new_labels[i, 3] = 1 if filtered_rank == 1 else 0

    torch.save(new_labels, OUT_PATH)

    n_pos = int((new_labels[:, 3] == 1).sum())
    n_neg = N - n_pos
    print(f"Saved → {OUT_PATH}")
    print(f"  Total queries : {N}")
    print(f"  Positive (rank=1) : {n_pos}  ({100*n_pos/N:.2f}%)")
    print(f"  Negative          : {n_neg}  ({100*n_neg/N:.2f}%)")
    print(f"  Positive rate     : {n_pos/N:.4f}")


if __name__ == "__main__":
    main()
