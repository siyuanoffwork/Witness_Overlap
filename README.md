<div align="center">

# 🧬 Witness Overlap

**Directional Provenance Inside Open-Weight Model Families**

Siyuan Li<sup>1,3</sup>, Haoxuan Zeng<sup>1</sup>, Xin Luo<sup>2</sup>, Fernando Jia<sup>1</sup>, Florence Li<sup>1</sup><br>
Zhengyang Geng<sup>3</sup>, Zico Kolter<sup>3</sup>, Tai Sing Lee<sup>3</sup>, Tianqin Li<sup>3</sup>

<sup>1</sup> ModelOS &nbsp; · &nbsp; <sup>2</sup> University of Michigan &nbsp; · &nbsp; <sup>3</sup> Carnegie Mellon University

<a href="https://neurips.cc/Conferences/2026">
  <img src="assets/neurips-2026.svg" height="32" alt="Accepted at NeurIPS 2026">
</a>
<a href="https://arxiv.org/pdf/2609.31784">
  <img src="assets/arxiv.svg" height="32" alt="Read the paper on arXiv: 2609.31784">
</a>

**Accepted at NeurIPS 2026**

[📄 Paper](https://arxiv.org/pdf/2609.31784) · [🚀 Quick Start](#quick-start) · [🔎 Tasks](#tasks) · [🗂️ Model Registry](#model-registry)

</div>

---

Witness Overlap studies **which checkpoint came first** within a family of related open-weight models. Given two candidate checkpoints, we use a third same-family checkpoint as a witness and compare the geometry of their weight updates to infer parent–child direction **without prompts or training**. The same witness scores support root identification and controlled-chain ordering, with a calibrated classifier for distinguishing parent–child pairs from siblings.

This repository provides the **LLM implementation** for four tasks:

| Task | Goal |
| :--- | :--- |
| 🌱 **Root identification** | Find the base checkpoint within a model family. |
| 🧭 **Parent–child direction** | Orient a known parent–child pair using one or more witnesses. |
| 🔀 **Pair discrimination** | Distinguish parent–child pairs from siblings using a calibrated classifier. |
| ⛓️ **Single-chain ordering** | Order sequentially fine-tuned checkpoints relative to a known base. |

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/siyuanoffwork/Witness_Overlap.git
cd Witness_Overlap
pip install -r requirements.txt
```

### 2. Set up your model directory

The registry stores HuggingFace repository ids and relative paths such as `models/...`, but no local absolute paths. The only path requirement is that `--models-root` points to a local directory under which those relative paths can be resolved. For example, if a registry path is `models/family/stage/repo`, then `--models-root /path/to/dataset-root` resolves it as `/path/to/dataset-root/models/family/stage/repo`.

You can also set:

```bash
export DATASET_ROOT=/path/to/dataset-root
```

### 3. Download models

Download the repositories to the registry paths. Adapter entries are merged into dense checkpoints automatically.

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

### 4. Compute witness overlap features

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

- `mean_overlap`: top-$k$ right-singular subspace overlap with $k=16$,
- `overlap_cos_signed`: signed Frobenius cosine.

For a quick smoke test on one family:

```bash
python -m witness_overlap.compute_features \
  --families gpt2 \
  --models-root /path/to/dataset-root \
  --out results/triplet_features_gpt2.json
```

<a id="tasks"></a>

## 🔎 Tasks

### Root identification

```bash
python -m witness_overlap.tasks.root_identification \
  --features results/triplet_features.json \
  --out results/root_identification.json
```

### Parent–child direction and $N_w$ sweep

```bash
python -m witness_overlap.tasks.parent_child_direction \
  --features results/triplet_features.json \
  --out results/parent_child_direction.json \
  --max-k 6
```

### Pair discrimination

```bash
python -m witness_overlap.tasks.pair_discrimination \
  --features results/triplet_features.json \
  --out results/pair_discrimination.json
```

This task uses a single-feature logistic regressor under leave-one-family-out evaluation. The decision threshold is selected on the training families only.

### Single-chain ordering

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

<a id="model-registry"></a>

## 🗂️ Model Registry

[`data/model_registry.json`](data/model_registry.json) lists the LLM checkpoints used in the paper.

| Families | Total checkpoints | Base | SFT descendants | RLHF / alignment descendants |
| :---: | :---: | :---: | :---: | :---: |
| **16** | **176** | **16** | **130** | **30** |

For the method, theoretical analysis, and full experimental results, see the [paper](https://arxiv.org/pdf/2609.31784).
