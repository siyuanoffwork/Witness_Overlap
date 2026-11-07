# Witness Overlap

This repository contains the minimal code used for the LLM Witness Overlap tasks in the paper:

1. root identification,
2. parent-child direction, including the \(N_w\)-witness sweep,
3. parent-child versus sibling pair discrimination,
4. single-chain order recovery.

## Data Registry

`data/model_registry.json` lists the 176 LLM checkpoints used by the paper. It contains 16 base checkpoints, 130 SFT descendants, and 30 RLHF or alignment descendants.

The registry stores HuggingFace repository ids and relative paths such as `models/...`, but no local absolute paths. The only path requirement is that `--models-root` points to a local directory under which those relative paths can be resolved. For example, if a registry path is `models/family/stage/repo`, then `--models-root /path/to/dataset-root` resolves it as `/path/to/dataset-root/models/family/stage/repo`.

You can also set:

```bash
export DATASET_ROOT=/path/to/dataset-root
```

## Download Models

After installing the dependencies below, download the repositories to the registry paths. Adapter entries are merged into dense checkpoints automatically.

```bash
export HF_TOKEN=...  # only needed for gated repositories

python -m witness_overlap.download_models \
  --registry data/model_registry.json \
  --models-root /path/to/dataset-root
```

Download one family first:

```bash
python -m witness_overlap.download_models \
  --families gpt2 \
  --models-root /path/to/dataset-root
```

## Install

```bash
pip install -r requirements.txt
```

## Step 1: Compute Witness Overlap Features

This computes anchored triplet features for each same-family triple:

```bash
python -m witness_overlap.compute_features \
  --registry data/model_registry.json \
  --models-root /path/to/dataset-root \
  --out results/triplet_features.json \
  --topk 16 \
  --seed 42
```

By default this computes both SVD overlap and signed Frobenius cosine. The output stores:

- `mean_overlap`: top-\(k\) right-singular subspace overlap with \(k=16\),
- `overlap_cos_signed`: signed Frobenius cosine.

For a quick smoke test on one family:

```bash
python -m witness_overlap.compute_features \
  --families gpt2 \
  --models-root /path/to/dataset-root \
  --out results/triplet_features_gpt2.json
```

## Task Scripts

Root identification:

```bash
python -m witness_overlap.tasks.root_identification \
  --features results/triplet_features.json \
  --out results/root_identification.json
```

Parent-child direction and \(N_w\) sweep:

```bash
python -m witness_overlap.tasks.parent_child_direction \
  --features results/triplet_features.json \
  --out results/parent_child_direction.json \
  --max-k 6
```

Pair discrimination:

```bash
python -m witness_overlap.tasks.pair_discrimination \
  --features results/triplet_features.json \
  --out results/pair_discrimination.json
```

This task uses a single-feature logistic regressor under leave-one-family-out evaluation. The decision threshold is selected on the training families only.

Single-chain order recovery:

```bash
python -m witness_overlap.tasks.single_chain_order \
  --chain-paths \
    base=/path/to/base/checkpoint \
    stage0=/path/to/stage0/checkpoint \
    stage1=/path/to/stage1/checkpoint \
    stage2=/path/to/stage2/checkpoint \
  --out results/single_chain_order.json
```

For the chain script, the first path is treated as the base. Each post-base checkpoint is used as an anchor only in comparisons of the form `anchor: base|other_stage`, then stages are sorted by the mean of those base-fixed overlaps.
