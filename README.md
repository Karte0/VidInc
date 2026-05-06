# VidInc: A Feature-Level Benchmark Suite for Video Incremental Learning under Category, Domain, and Distributional Shifts

> **NeurIPS 2026** | [Paper] | [Poster]

## Catalogue

* [Dataset Statistics](#dataset-statistics)
* [1. Getting Started](#1-getting-started)
* [2. Training and Evaluation](#2-training-and-evaluation)
* [3. Temporal Generator](#3-temporal-generator)
## Dataset Statistics

### VidInc‑Class Task Partitions

| Source Dataset | Total Categories | Categories / Task | Avg. Samples / Task | Total Videos | Training | Validation | Testing |
| -------------- | ---------------- | ----------------- | ------------------- | ------------ | -------- | ---------- | ------- |
| FCVID          | 239              | 24 (tasks 1‑9), 23 (task 10) | 4,478 | 88,898 | 44,475 | 22,219 | 22,204 |
| ActivityNet    | 200              | 20                | 1,002               | 14,948       | 10,023   | 2,512      | 2,413   |
| UCF101         | 101              | 11 (task 1), 10 (tasks 2‑10) | 786           | 13,320       | 7,856    | 2,664      | 2,800   |

### VidInc‑CrossDomain

| Domain Source | Total Videos | Training | Validation | Testing | Proportion |
| ------------- | ------------ | -------- | ---------- | ------- | ---------- |
| FCVID         | 63,491       | 31,748   | 15,874     | 15,869  | 74.8%      |
| ActivityNet   | 10,750       | 7,157    | 1,831      | 1,762   | 12.6%      |
| UCF101        | 10,759       | 6,345    | 2,152      | 2,262   | 12.6%      |
| **Total**     | **85,000**   | **45,250**| **19,857** | **19,893**| **100.0%** |

### VidInc‑Synthetic

| Domain ID          | Training | Validation | Testing |
| ------------------ | -------- | ---------- | ------- |
| ActivityNet (Source)| 10,023   | 2,413      | 2,512   |
| syn1               | 10,023   | 2,413      | 2,512   |
| syn2               | 10,023   | 2,413      | 2,512   |
| syn3               | 10,023   | 2,413      | 2,512   |

### Overall Statistical Profile

| Property            | VidInc‑Class | VidInc‑CrossDomain | VidInc‑Synthetic |
| ------------------- | ------------ | ------------------ | ---------------- |
| Source datasets     | FCVID, ActivityNet, UCF101 | FCVID, ActivityNet, UCF101 | ActivityNet |
| Categories          | 101 / 200 / 239 (per dataset) | 43 (shared)        | 200              |
| Increments          | 10 tasks     | 3 domains          | 1 source + 3 synthetic |
| Total videos        | 88,898 / 14,948 / 13,320 | 85,000             | 59,792           |
| Sample partition    | Balanced across tasks | Imbalanced (12.6%–74.8%) | Identical across domains |
| Domain gap type     | Semantic     | Natural (cross‑dataset) | Synthetic (controlled) |
| Avg. pairwise MMD   | N/A          | 0.12–0.58 (per category) | 0.9069           |

## 1. Getting Started

### 1.1 Environment Setup

```bash
conda create -n VidInc python=3.10.13
conda activate VidInc
conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 1.2 Download Datasets

CLIP features of FCVID, Activity and UCF101 are provided by us.
VidInc-Class and VidInc-CrossDomain datasets are subsets partitioned from the above three datasets.
VidInc-Synthetic datasets are synthesized and provided by us.

You can download all datasets and partition files from Baidu Disk or HuggingFace:

| Dataset          | Description                                                  | Link                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FCVID            | CLIP ViT-L/14 features, 239 classes                          | [Baidu Disk](https://pan.baidu.com/s/1CPpbjaU5RPHB5sLj_m4HHQ?pwd=qnd8) |
| ActivityNet      | CLIP ViT-L/14 features, 200 classes                          | [Baidu Disk](https://pan.baidu.com/s/1P_w3iSN3spNbHX105KFnRA?pwd=yafc) |
| UCF101           | CLIP ViT-L/14 features, 101 classes                          | [Baidu Disk](https://pan.baidu.com/s/13M0t-ttdE_GRh4jApfrrsw?pwd=qhb9) |
| VidInc-Synthetic | ActivityNet + 3 synthetic perturbation domains               | [Baidu Disk](https://pan.baidu.com/s/1El-8FvaVAE0zTC4TkS3O2w?pwd=xiti) |
| VidInc           | All datasets(CLIP ViT-L/14 features) and partition           | [Hugging Face](https://huggingface.co/datasets/Karte0/VidInc) |
| VidInc-sample    | A representative sample of datasets(CLIP ViT-L/14 features) and partition | [Hugging Face](https://huggingface.co/datasets/Karte0/VidInc_sample) |

## 2. Training and Evaluation

We provide three separate run scripts for the three main incremental settings:

| Script                   | Setting            | Datasets                                 |
| ------------------------ | ------------------ | ---------------------------------------- |
| `compute_acc_bwt.py`     | VidInc-Class       | FCVID / ActivityNet / UCF101             |
| `compute_acc_bwt_dil.py` | VidInc-CrossDomain | FCVID-sub / ActivityNet-sub / UCF101-sub |
| `compute_acc_bwt_syn.py` | VidInc-Synthetic   | ActivityNet + Syn1/2/3                   |

### 2.1  Configure Parameters

All three scripts share the same training hyper‑parameters and Transformer settings.
Modify the variables at the top of each file before running. Key parameters are listed below: 

| Parameter             | Description                                                  |
| :-------------------- | :----------------------------------------------------------- |
| `DEVICE`              | Training device (`e.g., 'cuda:0'` )                          |
| `BATCH_SIZE`          | Batch size for training                                      |
| `LEARNING_RATE`       | Learning rate                                                |
| `MEMORY_SIZE`         | Replay buffer capacity (number of vectors)                   |
| `D_MODEL`, `NHEAD`, … | Transformer architecture (768, 8, 2 layers, 2048 dim, dropout 0.1, `'cls'` pooling) |

#### Path Configuration for `compute_acc_bwt.py` and `compute_acc_bwt_syn.py`

Both scripts read dataset paths from a dictionary named `DATASET_CONFIGS` at the top of the file.

- **`compute_acc_bwt.py`** (Class‑Incremental) and **`compute_acc_bwt_syn.py`** (Synthetic Domain‑Incremental):
  Point the values to your class‑incremental JSON files, both scripts use the **same JSON files**:

  python

  ```
  DATASET_CONFIGS = {
      'actnet': 'class_incremental/activityNet/Anet.json',
      'fcvid':  'class_incremental/fcvid/fcvid.json',
      'ucf':    'class_incremental/ucf101/ucf.json'
  }
  ```

#### Path Configuration for `compute_acc_bwt_dil.py`

This script uses separate variables for H5 feature files and split directories. Update them to match your data layout:

| Variable            | Description                                                  |
| :------------------ | :----------------------------------------------------------- |
| `H5_BASE_DIR`       | Base directory for the H5 feature files                      |
| `DOMAIN_H5_FILES`   | Dict mapping domain name → H5 file path (relative to `H5_BASE_DIR`) |
| `SPLIT_BASE`        | Root directory for domain split files                        |
| `DOMAIN_SPLIT_DIRS` | Subdirectory per domain under `SPLIT_BASE`                   |
| `DOMAIN_FILE_NAMES` | Train/val file names per domain                              |

Example default values (adjust as needed):

python

```
H5_BASE_DIR = ""
DOMAIN_H5_FILES = {
    "activitynet": "act_vit.h5",
    "ucf101":      "ucf_vit_25.h5",
    "fcvid":       "fcvid_vit_train.h5"
}
SPLIT_BASE = "/domain_incremental"
DOMAIN_SPLIT_DIRS = {
    "activitynet": "Activitynet",
    "ucf101": "ucf101",
    "fcvid": "fcvid"
}
DOMAIN_FILE_NAMES = {
    "activitynet": {"train": "train.txt", "val": "val.txt"},
    "ucf101":      {"train": "train.txt", "val": "val.txt"},
    "fcvid":       {"train": "fcv_train.txt", "val": "fcv_val.txt"}
}
```

> **Multi‑order evaluation**:
> The domain‑incremental scripts automatically evaluate **all 6 domain orders** (for natural domains) or the defined synthetic sequences. Results are saved as per‑order CSV files – no extra code changes are needed.

### 2.2 Run Training and Evaluation

```bash
# Class-Incremental (VidInc-Class)
python compute_acc_bwt.py

# Real-world Domain-Incremental (VidInc-CrossDomain)
python compute_acc_bwt_dil.py

# Synthetic Domain-Incremental (VidInc-Synthetic)
python compute_acc_bwt_syn.py
```

Reported metrics: **ACC**, **BWT**.

---

## 3. Temporal Generator

The Temporal Generator ( `src/generator/temporal_generator.py` ) creates synthetic domain‑incremental data for **VidInc‑Synthetic**. It learns to perturb video features at the frame level, generating new domains while preserving category‑level semantics. We provide the details of architecture, training objectives, optimization procedure, hyperparameters, and model configuration in the appendix.

**Input:**

- A single H5 file containing clip‑level feature sequences (shape: `[n_frames, 768]`), stored under `/{video_id}/vectors`.
- One or more text files that map video IDs to class labels.

**Output:**

- `Kn` H5 files (`--output_suffix`), each representing a novel synthetic domain.
  The file structure mirrors the input: `/{video_id}/vectors` → synthetic frame features of the same shape.
- A PyTorch checkpoint (`--save`) containing the generator, classifier, and metadata.

### 3.1 Usage

bash

```
python temporal_generator.py \
  --h5 /path/to/original_features.h5 \
  --txts /path/to/labels.txt [additional_label_files] \
  --output synthetic \
  --Kn 3 \
  --gpu 0
```

The command above will produce three H5 files: `synthetic_1.h5`, `synthetic_2.h5`, `synthetic_3.h5`.

### 3.2 Key Parameters

| Argument       | Default        | Description                                                  |
| :------------- | :------------- | :----------------------------------------------------------- |
| `--h5`         | **required**   | Path to the input H5 feature file (e.g., `act_vit.h5`).      |
| `--txts`       | **required**   | One or more label text files. Each line: `video_id,class_label` or `video_id,_,class_label`. |
| `--output`     | `synthetic.h5` | Prefix for the output synthetic H5 files (`prefix_1.h5`, …, `prefix_Kn.h5`). |
| `--Kn`         | `3`            | Number of novel domains to generate.                         |
| `--hidden`     | `768`          | Hidden dimension of the Transformer inside the generator.    |
| `--n_layers`   | `2`            | Number of Transformer encoder layers.                        |
| `--n_heads`    | `4`            | Number of attention heads.                                   |
| `--epochs`     | `25`           | Total training epochs (alternating G/F updates).             |
| `--pre_epochs` | `15`           | Pre‑training epochs for the semantic classifier (Yp).        |
| `--iters`      | `10`           | Number of G/F update iterations per epoch.                   |
| `--k_shot`     | `32`           | Number of samples per class in each inner step (must be ≤ available samples). |
| `--bs`         | `32`           | Batch size for classifier training (pretrain + F step).      |
| `--lr_g`       | `1e-4`         | Learning rate for the generator (G).                         |
| `--lr_f`       | `5e-4`         | Learning rate for the online classifier (F).                 |
| `--lam_d`      | `0.5`          | Weight for the OT‑based diversity/novelty loss.              |
| `--lam_c`      | `0.01`         | Weight for the cycle‑consistency loss.                       |
| `--lam_ce`     | `0.1`          | Weight for the cross‑entropy classification loss on synthetic data. |
| `--alpha`      | `0.5`          | Ratio of synthetic samples in the F classifier update (real vs. synthetic). |
| `--ot_reg`     | `0.5`          | Regularisation coefficient for the Sinkhorn OT distance.     |
| `--gpu`        | `3`            | GPU device index.                                            |
| `--seed`       | `42`           | Random seed.                                                 |
| `--save`       | `best3.pth`    | Path for the best model checkpoint.                          |

### 3.3 Training Overview

1. **Data loading** – Reads the H5 file and label texts. Videos are normalised frame‑wise.
2. **Pretraining** – A small MLP classifier (`Yp`) is trained for `--pre_epochs` epochs to capture class semantics; it remains frozen afterwards.
3. **Alternating optimisation** – Each epoch runs `--iters` steps:
   - **F classifier update** – An online classifier is trained on real features *and* generated synthetic samples (using an exponential moving average with ratio `--alpha`).
   - **G generator update** – For every class, the generator creates `Kn` synthetic versions. Losses include:
     - *Novelty*: OT distance between synthetic and real feature subsets.
     - *Diversity*: OT distance between different synthetic domains.
     - *Cycle‑consistency*: L1 loss when mapping a synthetic sample back to the source domain.
     - *Classification*: Cross‑entropy with the frozen `Yp`.
4. **Saving** – The checkpoint with the lowest F‑loss is saved to `--save`.

### 3.4 Output Data Structure

After training, the generator is used to synthesise the new domains. For each input video (identified by its directory name in the H5), `Kn` synthetic variants are written into separate H5 files. The internal structure is identical to the source:

text

```
synthetic_1.h5
 ├── video_A/vectors   # (T, 768) synthetic features
 ├── video_B/vectors
 ...

synthetic_2.h5
 ├── video_A/vectors
 ├── video_B/vectors
 ...
```

These H5 files can then be used directly as additional domains for VidInc‑Synthetic training (see `compute_acc_bwt_syn.py`).

> **Note:** The generator training is computationally intensive; for large datasets, consider reducing `--iters` or `--epochs` initially for a quick sanity check. The effective batch size for the generator updates is controlled by `--k_shot` – increase it if GPU memory permits to stabilise OT distances.


