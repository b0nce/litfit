import time
import warnings
from typing import Any, cast

import torch

from .datasets import encode_texts, load_askubuntu, split_data
from .device import DEVICE
from .dispatch import generate_all_projections
from .evaluation import evaluate_projections, evaluate_retrieval_fast, find_dim_range
from .stats import compute_all_stats, compute_stats

warnings.filterwarnings("ignore")

DATASETS = {
    'AskUbuntu': load_askubuntu,
    # 'Quora': load_quora,
}

MODELS = {
    # 'Qwen-0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    'E5-base-v2': 'intfloat/e5-base-v2',
}


def run_benchmark(
    datasets: dict | None = None,
    models: dict | None = None,
    max_groups: int = 1500,
    dim_fractions: tuple[float, ...] | None = None,
    include_neg: bool = True,
    seed: int = 42,
) -> dict:
    """Run full benchmark across datasets x models.

    All embeddings and projections stay on device (CUDA/CPU).

    Returns:
        dict of {(dataset, model): (results, summary)}
    """
    if datasets is None:
        datasets = DATASETS
    if models is None:
        models = MODELS

    print(f"Using device: {DEVICE}")
    all_results = {}

    for ds_name, ds_loader in datasets.items():
        print(f"\n{'='*80}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*80}")

        all_ids, all_texts, id_to_group = ds_loader(max_groups=max_groups, seed=seed)

        for model_name, model_path in models.items():
            print(f"\n{'─'*60}")
            print(f"  MODEL: {model_name} ({model_path})")
            print(f"{'─'*60}")

            embs = encode_texts(model_path, all_texts)

            print("  Splitting...")
            data = split_data(all_ids, all_texts, embs, id_to_group, seed=seed)

            train_ids, _, train_embs, _ = data['train']
            val_ids, _, val_embs, _ = data['val']
            test_ids, _, test_embs, _ = data['test']

            print("\n  Baseline (val):  ", end="")
            bl_val = evaluate_retrieval_fast(val_embs, val_ids, id_to_group)
            print(f"MAP={bl_val['MAP@50']:.4f}")
            print("  Baseline (test): ", end="")
            bl_test = evaluate_retrieval_fast(test_embs, test_ids, id_to_group)
            print(f"MAP={bl_test['MAP@50']:.4f}")

            print("\n  Computing statistics...")
            t0 = time.time()
            if include_neg:
                st, neg = compute_all_stats(train_embs, train_ids, id_to_group)
            else:
                st = compute_stats(train_embs, train_ids, id_to_group)
                neg = None
            print(f"    Stats computed in {time.time()-t0:.1f}s")

            if dim_fractions is None:
                print("\n  Finding best dim fractions...")
                t0 = time.time()
                dim_fractions_run = find_dim_range(
                    st,
                    val_embs,
                    val_ids,
                    id_to_group,
                    metric='MAP@50',
                )
                print(f"    Found {len(dim_fractions_run)} fractions in {time.time()-t0:.1f}s")
            else:
                dim_fractions_run = dim_fractions

            print("\n  Generating projections...")
            t0 = time.time()
            all_W: dict[Any, torch.Tensor] = cast(
                dict, generate_all_projections(st, neg, include_neg_methods=include_neg)
            )
            print(f"    Generated in {time.time()-t0:.1f}s")

            print(f"\n  Evaluating ({len(all_W)} projections)...")
            try:
                results, summary = evaluate_projections(
                    all_W,
                    val_embs,
                    val_ids,
                    id_to_group,
                    test_embs=test_embs,
                    test_ids=test_ids,
                    dim_fractions=dim_fractions_run,
                    metric='MAP@50',
                )
            except KeyboardInterrupt:
                print("  Interrupted, saving partial results...")
                results, summary = {}, {}

            all_results[(ds_name, model_name)] = (results, summary)

            if results:
                _print_top_configs(
                    results,
                    all_W,
                    bl_val,
                    bl_test,
                    test_embs,
                    test_ids,
                    id_to_group,
                    metric='MAP@50',
                )

    return all_results


def _print_top_configs(
    results: dict,
    all_W: dict,
    bl_val: dict,
    bl_test: dict,
    test_embs: torch.Tensor,
    test_ids: list,
    id_to_group: dict,
    metric: str = 'MAP@50',
    top_k: int = 10,
) -> None:
    """Print the top projection configs with improvement over baseline."""
    from .device import to_torch
    from .evaluation import evaluate_retrieval_fast

    val_base = bl_val[metric]
    test_base = bl_test[metric]

    # Flatten: (key, n_dims) → val_score
    flat = []
    for key, dim_dict in results.items():
        for n_dims, scores in dim_dict.items():
            flat.append((key, n_dims, scores[metric]))

    flat.sort(key=lambda x: x[2], reverse=True)
    # Deduplicate by key (keep best dim per config) then take top_k
    seen_keys = set()
    top = []
    for key, n_dims, val_score in flat:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        top.append((key, n_dims, val_score))
        if len(top) >= top_k:
            break

    # Evaluate top configs on test set
    test_embs_t = to_torch(test_embs)

    print(f"\n{'='*80}")
    print(f"  TOP {len(top)} CONFIGS (improvement over baseline)")
    print(f"  Baseline  val {metric}={val_base:.4f}  test {metric}={test_base:.4f}")
    print(f"{'='*80}")
    print(
        f"  {'#':<3s}  {'Method':<24s}  {'Dims':>5s}  " f"{'Val':>7s} {'Δval':>7s}  {'Test':>7s} {'Δtest':>7s}  Config"
    )
    print(f"  {'-'*90}")

    for i, (key, n_dims, val_score) in enumerate(top, 1):
        W = all_W[key]
        p = test_embs_t @ W
        if n_dims is not None:
            p = p[:, :n_dims]
        test_score = evaluate_retrieval_fast(p, test_ids, id_to_group)[metric]

        dims_str = str(n_dims) if n_dims is not None else 'full'
        cfg_str = ', '.join(key[1:]) if len(key) > 1 else ''
        val_delta = val_score - val_base
        test_delta = test_score - test_base

        print(
            f"  {i:<3d}  {key[0]:<24s}  {dims_str:>5s}  "
            f"{val_score:>7.4f} {val_delta:>+7.4f}  "
            f"{test_score:>7.4f} {test_delta:>+7.4f}  {cfg_str}"
        )


if __name__ == "__main__":
    all_results = run_benchmark(
        datasets=DATASETS,
        models=MODELS,
        max_groups=1500,
        include_neg=True,
    )
