"""Tests for UnionFind and split_data (no external dependencies needed)."""

import torch

from litfit.datasets import UnionFind, split_data
from litfit.device import DEVICE, DTYPE


class TestUnionFind:
    def test_singleton(self):
        uf = UnionFind()
        assert uf.find('a') == 'a'

    def test_union_two(self):
        uf = UnionFind()
        uf.union('a', 'b')
        assert uf.find('a') == uf.find('b')

    def test_transitive(self):
        uf = UnionFind()
        uf.union('a', 'b')
        uf.union('b', 'c')
        assert uf.find('a') == uf.find('c')

    def test_separate_groups(self):
        uf = UnionFind()
        uf.union('a', 'b')
        uf.union('c', 'd')
        assert uf.find('a') != uf.find('c')

    def test_multiple_unions_same_group(self):
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        # All should be in the same group
        root = uf.find(1)
        assert uf.find(2) == root
        assert uf.find(3) == root
        assert uf.find(4) == root

    def test_idempotent_union(self):
        uf = UnionFind()
        uf.union('a', 'b')
        root_before = uf.find('a')
        uf.union('a', 'b')  # union again
        assert uf.find('a') == root_before

    def test_integer_keys(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 3)
        assert uf.find(0) == uf.find(3)
        assert uf.find(1) == uf.find(2)


class TestSplitData:
    def test_split_sizes(self):
        n = 100
        d = 8
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // 5 for i in range(n)}  # 20 groups of 5

        result = split_data(ids, [f"text_{i}" for i in range(n)], embs, id_to_group)

        assert 'train' in result
        assert 'val' in result
        assert 'test' in result

        # All items should be accounted for
        total = sum(len(result[s][0]) for s in ['train', 'val', 'test'])
        assert total == n

    def test_group_integrity(self):
        """All items in a group should end up in the same split."""
        n = 50
        d = 4
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // 5 for i in range(n)}  # 10 groups of 5

        result = split_data(ids, [f"t{i}" for i in range(n)], embs, id_to_group)

        for split_name in ['train', 'val', 'test']:
            split_ids = result[split_name][0]
            split_groups = {id_to_group[qid] for qid in split_ids}
            # For each group in this split, ALL items of that group should be here
            for gid in split_groups:
                group_items = [i for i in range(n) if id_to_group[i] == gid]
                for item in group_items:
                    assert item in split_ids, f"Group {gid} split across sets"

    def test_no_overlap(self):
        n = 40
        d = 4
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // 4 for i in range(n)}  # 10 groups

        result = split_data(ids, [f"t{i}" for i in range(n)], embs, id_to_group)

        train_ids = set(result['train'][0])
        val_ids = set(result['val'][0])
        test_ids = set(result['test'][0])

        assert not train_ids & val_ids
        assert not train_ids & test_ids
        assert not val_ids & test_ids

    def test_embeddings_match_ids(self):
        n = 20
        d = 4
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // 5 for i in range(n)}

        result = split_data(ids, [f"t{i}" for i in range(n)], embs, id_to_group)

        for split_name in ['train', 'val', 'test']:
            split_ids, _, split_embs, _ = result[split_name]
            assert split_embs.shape[0] == len(split_ids)
            assert split_embs.shape[1] == d

    def test_custom_fractions(self):
        n = 50
        d = 4
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // 5 for i in range(n)}  # 10 groups

        result = split_data(
            ids,
            [f"t{i}" for i in range(n)],
            embs,
            id_to_group,
            train_frac=0.5,
            val_frac=0.3,
        )

        total = sum(len(result[s][0]) for s in ['train', 'val', 'test'])
        assert total == n
