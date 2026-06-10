"""
KGDataset class and DataLoader builder for Knowledge Graph link prediction.
Supports FB15k-237, WN18RR, and Hetionet.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA, NBFNET_CONFIG, RANDOM_SEED

logger = logging.getLogger(__name__)


class KGDataset(Dataset):
    """
    Knowledge Graph dataset for link prediction.

    Provides triples (head, relation, tail) plus per-edge provenance weights.
    Supports negative sampling for training.

    Args:
        dataset_name: One of "fb15k237", "wn18rr", "hetionet".
        split: One of "train", "valid", "test".
        num_negative_samples: Number of negatives per positive (training only).
        device: Torch device to load tensors onto.
        use_provenance: Whether to load provenance weights.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        num_negative_samples: int = None,
        device: str = "cpu",
        use_provenance: bool = True,
    ):
        super().__init__()
        assert dataset_name in ("fb15k237", "wn18rr", "hetionet"), (
            f"Unknown dataset: {dataset_name}"
        )
        assert split in ("train", "valid", "test"), (
            f"Unknown split: {split}"
        )

        self.dataset_name = dataset_name
        self.split = split
        self.num_negative_samples = (
            num_negative_samples
            if num_negative_samples is not None
            else NBFNET_CONFIG["num_negative_samples"]
        )
        self.device = device
        self.use_provenance = use_provenance

        data_dir = Path(PROCESSED_DATA[dataset_name])
        self._load_metadata(data_dir)
        self._load_triples(data_dir, split)
        self._load_provenance(data_dir)

        # Build all_triples set for filtered negative sampling
        self._build_true_triples_set(data_dir)

    def _load_metadata(self, data_dir: Path):
        stats_path = data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.num_entities = stats["num_entities"]
            self.num_relations = stats["num_relations"]
        else:
            # Fallback: infer from triples
            train = torch.load(data_dir / "triples_train.pt", weights_only=True)
            self.num_entities = int(train[:, [0, 2]].max().item()) + 1
            self.num_relations = int(train[:, 1].max().item()) + 1

        # Load entity/relation mappings
        e2id_path = data_dir / "entity2id.json"
        r2id_path = data_dir / "relation2id.json"
        if e2id_path.exists():
            with open(e2id_path) as f:
                self.entity2id = json.load(f)
            self.id2entity = {v: k for k, v in self.entity2id.items()}
        else:
            self.entity2id = {}
            self.id2entity = {}

        if r2id_path.exists():
            with open(r2id_path) as f:
                self.relation2id = json.load(f)
            self.id2relation = {v: k for k, v in self.relation2id.items()}
        else:
            self.relation2id = {}
            self.id2relation = {}

    def _load_triples(self, data_dir: Path, split: str):
        path = data_dir / f"triples_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Triples not found at {path}. Run scripts/download_datasets.py first."
            )
        self.triples = torch.load(path, weights_only=True).long()
        logger.info(
            f"Loaded {len(self.triples)} {split} triples for {self.dataset_name}"
        )

    def _load_provenance(self, data_dir: Path):
        prov_path = data_dir / "provenance_weights.pt"
        if self.use_provenance and prov_path.exists():
            all_prov = torch.load(prov_path, weights_only=True).float()
            # Provenance is indexed by edge position in training set only
            if self.split == "train" and len(all_prov) == len(self.triples):
                self.provenance_weights = all_prov
            else:
                # For valid/test, use default weight
                self.provenance_weights = torch.ones(len(self.triples))
        else:
            self.provenance_weights = torch.ones(len(self.triples))

    def _build_true_triples_set(self, data_dir: Path):
        """Build set of all known true triples for filtered evaluation."""
        self.true_triples = set()
        for sp in ("train", "valid", "test"):
            path = data_dir / f"triples_{sp}.pt"
            if path.exists():
                t = torch.load(path, weights_only=True).long()
                for row in t:
                    h, r, tl = row[0].item(), row[1].item(), row[2].item()
                    self.true_triples.add((h, r, tl))

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict:
        h, r, t = self.triples[idx]
        h, r, t = h.item(), r.item(), t.item()
        prov = self.provenance_weights[idx].item()

        item = {
            "head": h,
            "relation": r,
            "tail": t,
            "provenance_weight": prov,
            "query_idx": idx,
        }

        if self.split == "train":
            item["negatives"] = self._sample_negatives(h, r, t)

        return item

    def _sample_negatives(self, h: int, r: int, t: int) -> torch.Tensor:
        """Sample negative tails, avoiding known true triples."""
        negatives = []
        attempts = 0
        max_attempts = self.num_negative_samples * 10

        while len(negatives) < self.num_negative_samples and attempts < max_attempts:
            neg_t = torch.randint(0, self.num_entities, (1,)).item()
            if (h, r, neg_t) not in self.true_triples and neg_t != t:
                negatives.append(neg_t)
            attempts += 1

        # Pad if needed (shouldn't happen for large KGs)
        while len(negatives) < self.num_negative_samples:
            negatives.append(torch.randint(0, self.num_entities, (1,)).item())

        return torch.tensor(negatives[:self.num_negative_samples], dtype=torch.long)

    def get_all_triples_tensor(self) -> torch.Tensor:
        """Return all triples as (N, 3) tensor."""
        return self.triples

    def get_all_provenance(self) -> torch.Tensor:
        """Return all provenance weights as (N,) tensor."""
        return self.provenance_weights


def kg_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for KGDataset."""
    result = {
        "head": torch.tensor([b["head"] for b in batch], dtype=torch.long),
        "relation": torch.tensor([b["relation"] for b in batch], dtype=torch.long),
        "tail": torch.tensor([b["tail"] for b in batch], dtype=torch.long),
        "provenance_weight": torch.tensor(
            [b["provenance_weight"] for b in batch], dtype=torch.float
        ),
        "query_idx": torch.tensor([b["query_idx"] for b in batch], dtype=torch.long),
    }
    if "negatives" in batch[0]:
        result["negatives"] = torch.stack([b["negatives"] for b in batch])
    return result


def build_dataloaders(
    dataset_name: str,
    batch_size: int = None,
    num_workers: int = 4,
    device: str = "cpu",
    use_provenance: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, KGDataset]:
    """
    Build train, valid, test DataLoaders for a dataset.

    Returns:
        (train_loader, valid_loader, test_loader, train_dataset)
        The train_dataset is returned for metadata access (num_entities, etc.).
    """
    batch_size = batch_size or NBFNET_CONFIG["batch_size"]

    train_ds = KGDataset(
        dataset_name, "train",
        num_negative_samples=NBFNET_CONFIG["num_negative_samples"],
        device=device,
        use_provenance=use_provenance,
    )
    valid_ds = KGDataset(
        dataset_name, "valid",
        device=device,
        use_provenance=use_provenance,
    )
    test_ds = KGDataset(
        dataset_name, "test",
        device=device,
        use_provenance=use_provenance,
    )

    # Share true triples from train for filtered evaluation
    valid_ds.true_triples = train_ds.true_triples | valid_ds.true_triples
    test_ds.true_triples = train_ds.true_triples | valid_ds.true_triples | test_ds.true_triples

    torch.manual_seed(RANDOM_SEED)

    _pin = NBFNET_CONFIG["pin_memory"]
    _prefetch = NBFNET_CONFIG["prefetch_factor"] if num_workers > 0 else None
    _persist = NBFNET_CONFIG["persistent_workers"] and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=kg_collate_fn,
        pin_memory=_pin,
        prefetch_factor=_prefetch,
        persistent_workers=_persist,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=kg_collate_fn,
        pin_memory=_pin,
        prefetch_factor=_prefetch,
        persistent_workers=_persist,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=kg_collate_fn,
        pin_memory=_pin,
        prefetch_factor=_prefetch,
        persistent_workers=_persist,
    )

    return train_loader, valid_loader, test_loader, train_ds
