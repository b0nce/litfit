import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

from .device import DEVICE, DTYPE

try:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    _OPTIONAL_DEPS_AVAILABLE = True
except ImportError:
    _OPTIONAL_DEPS_AVAILABLE = False


class UnionFind:
    """Path-compressed union-find for grouping duplicate texts by shared positive pairs."""

    def __init__(self) -> None:
        self.parent: Dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _subsample_groups(
    groups: Dict[Any, List],
    id_to_text: Dict[Any, str],
    max_groups: int,
    max_per_group: int,
    seed: int,
    label: str,
) -> Tuple[List, List[str], Dict]:
    rng = random.Random(seed)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    group_ids = group_ids[:max_groups]

    all_ids, all_texts, id_to_group = [], [], {}
    for gid in group_ids:
        for qid in groups[gid][:max_per_group]:
            all_ids.append(qid)
            all_texts.append(id_to_text[qid])
            id_to_group[qid] = gid

    print(f"  {label}: {len(all_ids)} items, {len(group_ids)} groups")
    return all_ids, all_texts, id_to_group


def load_askubuntu(
    max_groups: int = 2000,
    max_per_group: int = 6,
    seed: int = 42,
) -> Tuple[List, List[str], Dict]:
    """AskUbuntu duplicate questions -- clear group structure."""
    if not _OPTIONAL_DEPS_AVAILABLE:
        raise ImportError("Install optional dependencies: pip install sentence-transformers datasets")
    print("Loading AskUbuntu...")
    ds = load_dataset("sentence-transformers/askubuntu", split="train", trust_remote_code=True)

    uf = UnionFind()
    texts_seen = {}
    id_to_text = {}
    row_id = 0

    for row in ds:
        query = row['query'].strip()
        positives = row.get('positive', []) or []

        if query not in texts_seen:
            texts_seen[query] = f"au_{row_id}"
            row_id += 1
        qid = texts_seen[query]
        id_to_text[qid] = query

        for pos_text in positives:
            pos_text = pos_text.strip()
            if not pos_text:
                continue
            if pos_text not in texts_seen:
                texts_seen[pos_text] = f"au_{row_id}"
                row_id += 1
            pid = texts_seen[pos_text]
            id_to_text[pid] = pos_text
            uf.union(qid, pid)

    groups = defaultdict(set)
    for qid in id_to_text:
        groups[uf.find(qid)].add(qid)
    groups = {k: list(v) for k, v in groups.items() if len(v) >= 2}

    return _subsample_groups(groups, id_to_text, max_groups, max_per_group, seed, "AskUbuntu")


def load_twitter_url(
    max_groups: int = 2000,
    max_per_group: int = 6,
    seed: int = 42,
) -> Tuple[List, List[str], Dict]:
    """Twitter URL paraphrase -- tweet pairs about same URL."""
    if not _OPTIONAL_DEPS_AVAILABLE:
        raise ImportError("Install optional dependencies: pip install sentence-transformers datasets")
    print("Loading TwitterURL...")
    ds = load_dataset("sentence-transformers/twitter-url-corpus", split="train", trust_remote_code=True)

    uf = UnionFind()
    texts_seen = {}
    pair_id = 0
    for row in ds:
        s1, s2 = row['sentence1'].strip(), row['sentence2'].strip()
        if not s1 or not s2:
            continue
        if s1 not in texts_seen:
            texts_seen[s1] = f"tw_{pair_id}_0"
        if s2 not in texts_seen:
            texts_seen[s2] = f"tw_{pair_id}_1"
        pair_id += 1
        uf.union(texts_seen[s1], texts_seen[s2])

    id_to_text = {v: k for k, v in texts_seen.items()}
    groups = defaultdict(set)
    for qid in id_to_text:
        groups[uf.find(qid)].add(qid)
    groups = {k: list(v) for k, v in groups.items() if len(v) >= 2}

    return _subsample_groups(groups, id_to_text, max_groups, max_per_group, seed, "TwitterURL")


def load_quora(
    max_groups: int = 2000,
    max_per_group: int = 6,
    seed: int = 42,
) -> Tuple[List, List[str], Dict]:
    """Quora duplicate questions."""
    if not _OPTIONAL_DEPS_AVAILABLE:
        raise ImportError("Install optional dependencies: pip install sentence-transformers datasets")
    print("Loading Quora...")
    ds = load_dataset("quora", split="train", trust_remote_code=True)

    uf = UnionFind()
    id_to_text = {}
    for row in ds:
        q1_id, q2_id = row['questions']['id']
        q1_text, q2_text = row['questions']['text']
        id_to_text[q1_id] = q1_text
        id_to_text[q2_id] = q2_text
        if row['is_duplicate']:
            uf.union(q1_id, q2_id)

    groups = defaultdict(set)
    for qid in id_to_text:
        groups[uf.find(qid)].add(qid)
    groups = {k: list(v) for k, v in groups.items() if len(v) >= 2}

    return _subsample_groups(groups, id_to_text, max_groups, max_per_group, seed, "Quora")


def encode_texts(
    model_name: str,
    texts: List[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode texts with a sentence-transformers model, return as torch tensor on device."""
    if not _OPTIONAL_DEPS_AVAILABLE:
        raise ImportError("Install optional dependencies: pip install sentence-transformers datasets")
    print(f"  Encoding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return torch.tensor(embs, device=DEVICE, dtype=DTYPE)


def split_data(
    all_ids: List,
    all_texts: List[str],
    embs: torch.Tensor,
    id_to_group: Dict,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, Tuple]:
    """Group-aware split into train/val/test.

    Returns:
        dict with keys 'train', 'val', 'test', each containing
        (ids, texts, embs_tensor, id_to_group).
    """
    groups = list(set(id_to_group[qid] for qid in all_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    s1 = int(train_frac * n)
    s2 = int((train_frac + val_frac) * n)

    splits = {
        'train': set(groups[:s1]),
        'val': set(groups[s1:s2]),
        'test': set(groups[s2:]),
    }

    result = {}
    for split_name, group_set in splits.items():
        mask = torch.tensor([id_to_group[qid] in group_set for qid in all_ids], dtype=torch.bool, device=embs.device)
        s_ids = [all_ids[i] for i in range(len(all_ids)) if mask[i]]
        s_texts = [all_texts[i] for i in range(len(all_texts)) if mask[i]]
        s_embs = embs[mask]
        result[split_name] = (s_ids, s_texts, s_embs, id_to_group)

    for s, (ids, _, emb, _) in result.items():
        n_groups = len(set(id_to_group[qid] for qid in ids))
        print(f"    {s}: {len(ids)} items, {n_groups} groups, emb={tuple(emb.shape)}")

    return result
