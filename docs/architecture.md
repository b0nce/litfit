# Architecture

Visual overview of the `litfit` codebase. All diagrams use [Mermaid](https://mermaid.js.org/) syntax.

## Module Dependency Graph

```mermaid
graph TD
    device["device.py<br/><small>DEVICE, DTYPE, to_torch, to_numpy</small>"]
    stats["stats.py<br/><small>compute_stats, compute_all_stats</small>"]
    methods["methods.py<br/><small>18 projection methods</small>"]
    dispatch["dispatch.py<br/><small>generate_all_projections</small>"]
    evaluation["evaluation.py<br/><small>evaluate_projections</small>"]
    benchmark["benchmark.py<br/><small>run_benchmark</small>"]
    datasets["datasets.py<br/><small>load_*, encode_texts</small>"]
    triton["triton_kernels.py<br/><small>fused covariance kernel</small>"]

    device --> stats
    device --> methods
    device --> datasets
    stats --> dispatch
    methods --> dispatch
    dispatch --> evaluation
    evaluation --> benchmark
    datasets --> benchmark
    triton -.->|optional| stats
```

## Data Pipeline Flow

```mermaid
flowchart LR
    subgraph Load
        A[load_askubuntu<br/>load_twitter_url<br/>load_quora]
    end

    subgraph Encode
        B[encode_texts]
    end

    subgraph Split
        C[split_data]
    end

    subgraph Stats
        D[compute_stats /<br/>compute_stats_streaming]
        E[compute_all_stats]
    end

    subgraph Project
        F[generate_fast_projections /<br/>generate_all_projections]
    end

    subgraph Evaluate
        G[find_dim_range]
        H[evaluate_projections]
    end

    A -- "ids, texts,<br/>id_to_group" --> B
    B -- "embeddings" --> C
    C -- "train / val / test<br/>splits" --> D
    D -- "st dict" --> E
    E -- "st, neg" --> F
    F -- "all_W dict" --> H
    C -- "val_embs, val_ids" --> G
    G -- "dim_fractions" --> H
    H -- "results, summary" --> I[Best projections]
```

## Method Taxonomy

```mermaid
graph TD
    root["Projection Methods<br/>(18 total)"]

    basic["Basic"]
    asym["Asymmetric"]
    iter["Iterative"]
    rank["Rank-reduced"]
    uber["Uber (hybrid)"]
    contr["Contrastive<br/><em>needs neg</em>"]
    resid["Residual<br/><em>needs neg</em>"]

    root --> basic
    root --> asym
    root --> iter
    root --> rank
    root --> uber
    root --> contr
    root --> resid

    basic --- b1["m_rayleigh"]
    basic --- b2["m_mse"]
    basic --- b3["m_ray_mse"]

    asym --- a1["m_asym_rayleigh"]
    asym --- a2["m_asym_ray_mse"]
    asym --- a3["m_ray_asym_refine"]
    asym --- a4["m_ray_mse_asym_refine"]

    iter --- i1["m_ray_iterate"]
    iter --- i2["m_ray_asym_refine_mse"]

    rank --- r1["m_split_rank_ray"]
    rank --- r2["m_split_rank_ray_mse"]
    rank --- r3["m_split_rank_ray_iterate"]

    uber --- u1["m_uber → (W, bias)"]

    contr --- c1["m_ray_contr_mse"]
    contr --- c2["m_ray_contr_mse_neg"]

    resid --- d1["m_resid_guided"]

    uber --- u2["m_uber_neg → (W, bias)<br/><em>needs neg</em>"]
    uber --- u3["m_uber_contr → (W, bias)<br/><em>needs neg</em>"]
```

## Evaluate Projections Loop

```mermaid
flowchart TD
    start([Start]) --> init["Initialize scores for all<br/>projection configs"]
    init --> pick{"Pick next config:<br/>explore or exploit?"}

    pick -- "explore<br/>(probability = explore_fraction)" --> sample["Sample config uniformly<br/>at random"]
    pick -- "exploit<br/>(probability = 1 - explore_fraction)" --> softmax["Sample config via softmax<br/>over current scores"]

    sample --> compute["Compute W = method(st, neg)"]
    softmax --> compute

    compute --> dims["For each dim_fraction:<br/>W_trunc = W[:, :k]"]
    dims --> eval["evaluate_retrieval_fast<br/>on val set"]
    eval --> update["Update best score<br/>for this config"]
    update --> check{More configs<br/>to try?}

    check -- Yes --> pick
    check -- No --> result([Return results, summary])

    interrupt["KeyboardInterrupt"] -.->|early stop| result
```
