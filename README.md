# VidInc: A Feature-Level Benchmark Suite for Video Incremental Learning under Category, Domain, and Distributional Shifts

> **ICASSP 2027** | [Paper] | [Poster]

## Catalogue

* [Dataset Statistics](#dataset-statistics)
* [1. Getting Started](#1-getting-started)
* [2. Training and Evaluation](#2-training-and-evaluation)
* [3. Temporal Generator](#3-temporal-generator)
* [Appendix](#appendix)
  * [1. Detailed Category Mapping for VidInc-CrossDomain](#1-detailed-category-mapping-for-vidinc-crossdomain)
  * [2. Temporal Generator Details](#2-temporal-generator-details)
  * [3. Experimental Setups](#3-experimental-setups)
  * [4. Per-Order Breakdown for VidInc-CrossDomain](#4-per-order-breakdown-for-vidinc-crossdomain)

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

### 2.1 Configure Parameters

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

```python
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

```python
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

The Temporal Generator (`src/generator/temporal_generator.py`) creates synthetic domain‑incremental data for **VidInc‑Synthetic**. It learns to perturb video features at the frame level, generating new domains while preserving category‑level semantics. We provide the details of architecture, training objectives, optimization procedure, hyperparameters, and model configuration in the appendix.

**Input:**

- A single H5 file containing clip‑level feature sequences (shape: `[n_frames, 768]`), stored under `/{video_id}/vectors`.
- One or more text files that map video IDs to class labels.

**Output:**

- `Kn` H5 files (`--output_suffix`), each representing a novel synthetic domain.
  The file structure mirrors the input: `/{video_id}/vectors` → synthetic frame features of the same shape.
- A PyTorch checkpoint (`--save`) containing the generator, classifier, and metadata.

### 3.1 Usage

```bash
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

```text
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

---

# Appendix

## 1. Detailed Category Mapping for VidInc-CrossDomain

Table 1.1 provides the exhaustive mapping between the **Unified Category** and the original labels in ActivityNet, UCF101, and FCVID.

**Table 1.1. Category mapping for VidInc-CrossDomain.**

| Unified Category | ActivityNet | UCF101 | FCVID |
|---|---|---|---|
| Archery | Archery | Archery | archery |
| Bowling | Playing ten pins | Bowling | bowling |
| Fencing | Doing fencing | Fencing | fencing |
| Rope Skipping | Rope skipping | JumpRope | ropeSkipping |
| Table Tennis | Ping-pong | TableTennisShot | tableTennis |
| Tai Chi | Tai chi | TaiChi | taiChiChuan |
| Boxing | Doing kickboxing | BoxingPunchingBag, BoxingSpeedBag | boxing, punchingBagWorkout |
| Cycling & Biking | Assembling bicycle, BMX, Fixing bicycle | Biking | assemblingABike, bikeTricks, biking |
| Extreme Aerial Sports | Bungee jumping, Powerbocking | SkyDiving | bungeeJumping, skydiving |
| Keyboard Instrument Performance | Playing piano | PlayingPiano | pianoPerformance |
| Skateboarding & Skating | Longboarding, Skateboarding | SkateBoarding | rollerSkating, skateboarding |
| Animal Riding & Competition | Camel ride, Calf roping, Horseback riding, Playing polo | HorseRiding, HorseRace | camel, horseRiding |
| Dance | Ballet, Belly dance, Breakdancing, Cumbia, Tango, Zumba | SalsaSpin, IceDancing | groupDance, socialDance, soloDance, weddingDance |
| Diving | Plataform diving, Springboard diving | Diving, CliffDiving | diving |
| Football / Soccer | Beach soccer, Futsal, Playing kickball | SoccerJuggling, SoccerPenalty | soccerAmateur, soccerProfessional |
| Frisbee Sports | Disc dog | FrisbeeCatch | playingFrisbeeWithDog, playingFrisbeeWithPeople |
| Paddle Water Sports | Canoeing, Kayaking, Rafting, River tubing | Kayaking, Rafting, Rowing | boating, rafting, river, rowing |
| Painting & Writing | Painting, Painting fence, Painting furniture | WritingOnBoard | doingGraffiti, painting, sprayPainting |
| Percussion Instrument Performance | Playing drums, Playing congas | Drumming, PlayingTabla | beatbox, rockBandPerformance |
| Rock Climbing & Rope Climbing | Rock climbing | RockClimbingIndoor, RopeClimbing | rockClimbing |
| Basketball | Layup drill in basketball | Basketball, BasketballDunk | basketballAmateur, basketballProfessional, wheelchairBasketball |
| Beverage Making | Making a lemonade, Mixing drinks | Mixing | makingCoffee, makingJuice, makingMilkTea, makingMixedDrinks, makingTea |
| Drill & Team Performance | Cheerleading, Baton twirling, Drum corps | BandMarching, MilitaryParade | chorus, marchingBand, parade, singingInKtv, singingOnStage |
| Oral & Facial Cleaning | Brushing teeth, Gargling mouthwash, Washing face, Washing hands | BrushingTeeth | brushingTeeth |
| Shaving & Body Care | Shaving, Shaving legs, Getting a tattoo, Getting a piercing | ShavingBeard | shavingBeard, tattooing |
| Surfing & Water Board Sports | Windsurfing, Surfing, Wakeboarding, Waterskiing | Surfing | kiteSurfing, surfing |
| Swimming | Swimming | BreastStroke, FrontCrawl | butterfly, swimmingAmateur, swimmingProfessional |
| Billiards & Board Games | Playing pool, Playing blackjack, Playing rubik cube, Rock-paper-scissors | Billiards | billiard, cardManipulation, playingChess, playingMahjong, solvingMagicCube |
| Hair Care | Blow-drying hair, Braiding hair, Brushing hair, Getting a haircut, Removing curlers | BlowDryHair, Haircut | hairCutting, hairstyleDesign |
| Ice & Snow Sports | Curling, Ice fishing, Shoveling snow, Snow tubing, Snowboarding, Skiing, Waxing skis | Skiing | iceSkating, makingSnowman, shovelingSnow, skiing, snowballFight |
| Makeup & Beauty Care | Applying sunscreen, Doing nails, Putting on makeup, Putting in contact lenses | ApplyEyeMakeup, ApplyLipstick | eyeMakeup, faceMassage, nailArtDesign, wearLipstick |
| Racket Sports | Playing racquetball, Playing squash, Playing badminton, Tennis serve with ball bouncing | TennisSwing | badminton, kickingShuttlecock, tennis, wheelchairTennis |
| Pet Care & Dog Walking | Clipping cat claws, Grooming dog, Grooming horse, Bathing dog, Walking the dog | WalkingWithDog | cat, dog, walkingWithDog |
| Combat & Martial Arts | Capoeira, Doing karate, Doing a powerbomb, Sumo, Arm wrestling | SumoWrestling, Punch, Nunchucks | armWrestling, playingWithNunChucks, streetFighting, sumoWrestling, taekwondo |
| Gymnastics | Tumbling, Using parallel bars, Using the balance beam, Using the pommel horse, Using uneven bars, Slacklining | ParallelBars, BalanceBeam, PommelHorse, UnevenBars, StillRings, FloorGymnastics, TrampolineJumping | hulaHoop, parkour, rhythmicGymnastics, yoga |
| Wind Instrument Performance | Playing accordion, Playing bagpipes, Playing flauta, Playing harmonica, Playing saxophone | PlayingFlute, PlayingDaf, PlayingDhol | accordionPerformance, flutePerformance, harmonicaPerformance, saxophonePerformance, trumpetPerformance |
| Festivals & Celebrations | Decorating the Christmas tree, Carving jack-o-lanterns, Wrapping presents | BlowingCandles | birthday, decoratingChristmasTree, fireworksShow, marriageProposal, weddingCeremony, weddingReception |
| String Instrument Performance | Playing violin, Playing guitarra | PlayingViolin, PlayingGuitar, PlayingCello, PlayingSitar | celloPerformance, chamberMusic, guitarPerformance, symphonyOrchestraPerformance, violinPerformance |
| Outdoor Recreation & Entertainment | Building sandcastles, Kite flying, Hitting a pinata, Hopscotch, Fun sliding down, Riding bumper cars, Swinging at the playground, Starting a campfire | Swing, JugglingBalls, YoYo | amusementPark, bumperCars, flyingKites, penSpinning, picnic, pitchingATent, playground, yoyoTricks |
| Strength Training & Fitness | Clean and jerk, Snatch, Doing crunches, Doing step aerobics, Elliptical trainer | CleanAndJerks, BenchPress, BodyWeightSquats, Lunges, PullUps, PushUps, WallPushups, HandstandPushups, HandstandWalking, JumpingJack | barbellWorkout, dumbbellWorkout, pullUps, pushUps, sitUps, treadmill |
| Cleaning & Housework | Cleaning shoes, Cleaning sink, Cleaning windows, Hand washing clothes, Ironing clothes, Mooping floor, Vacuuming floor, Washing dishes, Polishing forniture, Polishing shoes | MoppingFloor | carWashing, cleaningAppliance, cleaningCarpet, cleaningFloor, cleaningWindows, mowing, washingAnInfant, washingDishes |
| Knitting & Handicrafts | Knitting | Knitting | knitting, makingBookmark, makingBracelets, makingCeramicCraft, makingEarrings, makingFestivalCards, makingPaperFlowers, makingPaperPlane, makingPencilCases, makingPhoneCases, makingPhotoFrame, makingRings, makingShorts, makingWallet, paperCutting, sculpting |
| Cooking & Food Preparation | Making a cake, Making a sandwich, Making an omelette, Preparing pasta, Preparing salad, Baking cookies, Peeling potatoes, Sharpening knives | PizzaTossing, CuttingInKitchen | makingCake, makingChineseDumplings, makingCookies, makingEggTarts, makingFrenchFries, makingHotdog, makingIcecream, makingPizza, makingSalad, makingSandwich, makingSushi, roastingTurkey, deliciousFood, diningAtRestaurant, dinnerAtHome, groupBanquet |

---

## 2. Temporal Generator Details

This section provides a comprehensive description of the temporal generator $G_\theta$ introduced in the VidInc-Synthetic construction. We detail its architecture, training objectives, optimization procedure, hyperparameters, and model configuration.

### 2.1 Architecture of $G_\theta$

The generator operates on frame-level feature sequences. Given a normalized video clip $\mathbf{X}\in\mathbb{R}^{T\times D}$ with $T=25$ frames and $D=768$ (CLIP ViT-L/14 features), the forward pass for a target domain $k\in\{1,\dots,K_n\}$ ($K_n=3$) is defined by three successive operations:

$$
\mathbf{H} = \mathbf{X}\,\mathbf{W}_{\mathrm{proj}} \;\in \mathbb{R}^{T\times H},
$$

$$
\mathbf{Z}^{(k)} = \mathrm{Transformer}\bigl([\mathbf{e}_k; \mathbf{H}]\bigr) \;\in \mathbb{R}^{(T+1)\times H},
$$

$$
\tilde{\mathbf{X}}^{(k)} = \mathbf{X} + \mathbf{Z}^{(k)}_{1:T,\,:}\,\mathbf{W}_{\mathrm{out}},
$$

where $H=768$ is the latent dimensionality. The domain index $k$ is the only conditioning input.

- **Latent Projection:** The linear layer $\mathbf{W}_{\mathrm{proj}}\in\mathbb{R}^{D\times H}$ maps each frame independently into the latent space.
- **Domain-Conditioned Encoding:** A learnable domain embedding $\mathbf{e}_k = \mathbf{E}[k]\in\mathbb{R}^{H}$ is taken from an embedding table $\mathbf{E}\in\mathbb{R}^{(1+K_n)\times H}$. The table is initialized with orthogonal rows scaled by $\sqrt{H}$ to ensure diverse, well-conditioned starting vectors. The embedded token is prepended to the frame sequence $\mathbf{H}$, and the resulting $(T+1)$ tokens are processed by a 2-layer Transformer encoder with Pre-Layer Normalization (Pre-LN) and $H=768$, $4$ attention heads. The Transformer captures long-range inter-frame dependencies under the influence of the domain token.
- **Residual Synthesis:** After the Transformer, the output at the domain-token position (first row of $\mathbf{Z}^{(k)}$) is discarded. Only the $T$ frame-token outputs are projected back to the original feature dimension via $\mathbf{W}_{\mathrm{out}}\in\mathbb{R}^{H\times D}$, and a residual connection adds the result to the input $\mathbf{X}$. The output projection weights are initialized to zero, ensuring that at the beginning of training the generator acts as identity and the training signal comes solely from the losses.

This formulation guarantees that $G_\theta$ learns an additive shift on the temporal manifold, preserving the underlying semantic skeleton of the original video.

### 2.2 Optimization Objectives

Training the generator involves a multi-objective functional together with an auxiliary classifier $F_\psi$ that assesses the quality of the generated features. In the following, $\mu(\cdot)$ denotes temporal mean pooling, producing a feature vector of dimension $D$ for each video.

#### 2.2.1 Generator Losses

1. **Distributional Divergence** ($\mathcal{L}_{nov},\mathcal{L}_{div}$). Both terms rely on the Gromov-Wasserstein distance (GWD) computed between video-level mean features. Given two sets of features $\mathbf{A}_1,\mathbf{A}_2$ and $\mathbf{B}_1,\mathbf{B}_2$, we define the Gromov inner product distance

$$
\mathrm{GED}(\mathbf{A}_1,\mathbf{A}_2; \mathbf{B}_1,\mathbf{B}_2)= 2\,\mathrm{OT}(\mathbf{A}_1,\mathbf{B}_1)-\mathrm{OT}(\mathbf{A}_1,\mathbf{A}_2)-\mathrm{OT}(\mathbf{B}_1,\mathbf{B}_2),
$$

where $\mathrm{OT}(\cdot,\cdot)$ is the entropic optimal transport distance (Sinkhorn algorithm) with regularization $\varepsilon=0.5$.

For a batch of size $B$, we split it into two halves of size $n=B/2$. Then, for each novel domain $k$, the novelty loss is

$$
\mathcal{L}_{nov}
= \sum_{k=1}^{K_n}
\mathrm{GED}\bigl(\mu(\tilde{\mathbf{X}}^{(k)}_1),\,\mu(\tilde{\mathbf{X}}^{(k)}_2);\;
\mu(\mathbf{X}_1),\,\mu(\mathbf{X}_2)\bigr),
$$

which encourages the synthetic domain distributions to move away from the source manifold. The diversity loss for pairs of synthetic domains is

$$
\mathcal{L}_{div}=\sum_{1\le i<j\le K_n}\mathrm{GED}\bigl(\mu(\tilde{\mathbf{X}}^{(i)}_1),\,\mu(\tilde{\mathbf{X}}^{(i)}_2);\;
\mu(\tilde{\mathbf{X}}^{(j)}_1),\,\mu(\tilde{\mathbf{X}}^{(j)}_2)\bigr),
$$

forcing the domains to occupy distinct regions of the feature space.

2. **Structural Integrity** ($\mathcal{L}_{cyc}$). Cycle-consistency is enforced by feeding the synthetic features back into the same $G_\theta$ conditioned on the source domain (domain index $0$) and requiring reconstruction of the original features:

$$
\mathcal{L}_{cyc} = \sum_{k=1}^{K_n}\bigl\|\,G_\theta\bigl(\tilde{\mathbf{X}}^{(k)},\, 0\bigr) - \mathbf{X}\,\bigr\|_1,
$$

where the generator input $\tilde{\mathbf{X}}^{(k)}$ is detached from the computation graph to avoid back-propagation through a nested generation.

3. **Semantic Anchor** ($\mathcal{L}_{ce}$). A frozen pre-trained classifier $\phi$ (architecture identical to $F_\psi$) ensures class-consistency:

$$
\mathcal{L}_{ce} = \sum_{k=1}^{K_n}
\mathrm{CE}\bigl(\phi(\mu(\tilde{\mathbf{X}}^{(k)})),\, \mathbf{y}\bigr),
$$

where $\mathbf{y}$ are the ground-truth labels of the source batch $\mathbf{X}$ and $\mathrm{CE}$ denotes the cross-entropy loss.

The total generator loss is a weighted combination:

$$
\mathcal{L}_G = -\lambda_d\bigl(\mathcal{L}_{nov} + \mathcal{L}_{div}\bigr) + \lambda_c\,\mathcal{L}_{cyc} + \lambda_{ce}\,\mathcal{L}_{ce},
$$

with $\lambda_d=0.5$, $\lambda_c=0.01$, and $\lambda_{ce}=0.1$.

#### 2.2.2 Auxiliary Classifier $F_\psi$ and Quality Assessment

An auxiliary classifier $F_\psi$ (a small MLP, see Table 2.2) is jointly trained to evaluate the fidelity of the generated features. During the update of $F_\psi$, the generator $G_\theta$ is frozen; the loss is a mixture of cross-entropy on real and synthetic mean features:

$$
\mathcal{L}_F = (1-\alpha)\,\mathrm{CE}\bigl(F_\psi(\mu(\mathbf{X})),\,\mathbf{y}\bigr) + \frac{\alpha}{K_n}\sum_{k=1}^{K_n} \mathrm{CE}\bigl(F_\psi(\mu(\tilde{\mathbf{X}}^{(k)})),\,\mathbf{y}\bigr),
$$

with mixing coefficient $\alpha=0.5$.

Importantly, $F_\psi$ does not provide gradients to $G_\theta$ during the generator update. Instead, its loss $\mathcal{L}_F$ is monitored and used to select the best checkpoint of the generator ($G_\theta$ and $F_\psi$ are saved together). This ensures that only generators producing features that are discriminatively consistent with the source distribution are retained.

### 2.3 Training Procedure

The training proceeds in epochs, each consisting of $I=10$ iterations. Within one iteration, we first update $F_\psi$ using a batch drawn from the full dataset, and then update $G_\theta$ by sequentially processing class-balanced mini-batches. The detailed algorithm is given as Algorithm 2.1.

For the generator update, we iterate over all classes that have at least $k_{\mathrm{shot}}=32$ samples. For each such class, a batch of size $B=32$ is drawn; the batch is halved to compute the OT-based losses. The total generator loss is computed, gradients are accumulated, and after all classes have been processed, the accumulated gradients are clipped and $G_\theta$ is updated. This class-wise accumulation ensures a balanced contribution from every class despite the uneven dataset sizes.

The semantic classifier $\phi$ is pre-trained for $15$ epochs on the mean features of the source dataset and then frozen. All optimizers are Adam: learning rate $10^{-4}$ for $G_\theta$ and $5\times10^{-4}$ for $F_\psi$, with weight decay $10^{-5}$. Training runs for $E=25$ epochs.

**Algorithm 2.1: Training of the Temporal Generator $G_\theta$**

$$
\begin{array}{l}
\hline
\textbf{Algorithm 2.1: Training of the Temporal Generator } G_\theta \\
\hline
\textbf{Require:}\ \text{Dataset } \mathcal{D} = \{(X_i, y_i)\}\ \text{of frame sequences}; \\
\quad \text{Number of novel domains } K_n;\ \text{frozen classifier } \phi; \\
\quad \text{Hyperparameters } \lambda_d, \lambda_c, \lambda_{ce}, \alpha, \eta_G, \eta_F; \\
\quad \text{Epochs } E;\ \text{iterations per epoch } I;\ \text{min samples per class } k_{\mathrm{shot}}. \\[4pt]
\textbf{1:}\ \text{Initialize } G_\theta\ \text{and } F_\psi;\ \text{let } \mu(\cdot)\ \text{denote temporal mean pooling}. \\
\textbf{2:}\ \text{Pretrain } \phi\ \text{on } \mu(X)\ \text{for 15 epochs; freeze } \phi. \\
\textbf{3:}\ \textbf{for}\ e = 1\ \textbf{to}\ E\ \textbf{do} \\
\textbf{4:}\ \quad \textbf{for}\ t = 1\ \textbf{to}\ I\ \textbf{do} \\
\textbf{5:}\ \quad\quad \text{// Stage 1: update } F_\psi\ \text{(with } G_\theta\ \text{frozen)} \\
\textbf{6:}\ \quad\quad \text{Sample a batch } \mathcal{B}_F = \{(X, y)\}\ \text{from } \mathcal{D}; \\
\textbf{7:}\ \quad\quad \tilde{X}^{(k)} = G_\theta(X, k)\ \forall k \in \{1, \ldots, K_n\}\ \text{(no gradient)}; \\
\textbf{8:}\ \quad\quad \mathcal{L}_F = (1-\alpha)\,\mathrm{CE}\bigl(F_\psi(\mu(X)),\, y\bigr) + \tfrac{\alpha}{K_n} \sum_{k=1}^{K_n} \mathrm{CE}\bigl(F_\psi(\mu(\tilde{X}^{(k)})),\, y\bigr); \\
\textbf{9:}\ \quad\quad F_\psi \leftarrow F_\psi - \eta_F \nabla_{\psi} \mathcal{L}_F; \\
\textbf{10:}\ \quad\quad \text{// Stage 2: update } G_\theta\ \text{(with } F_\psi\ \text{frozen)} \\
\textbf{11:}\ \quad\quad g \leftarrow 0;\ \mathcal{C}_{\mathrm{valid}} = \{c : |\mathcal{D}_c| \ge k_{\mathrm{shot}}\}; \\
\textbf{12:}\ \quad\quad \textbf{for}\ c \in \mathcal{C}_{\mathrm{valid}}\ \textbf{do} \\
\textbf{13:}\ \quad\quad\quad \text{Sample batch } \mathcal{B}_c = \{(X, c)\}\ \text{of size } B;\ \text{split into } X_1, X_2; \\
\textbf{14:}\ \quad\quad\quad \tilde{X}^{(k)} = G_\theta(X, k)\ \forall k \in \{1, \ldots, K_n\}; \\
\textbf{15:}\ \quad\quad\quad \text{Compute } \mathcal{L}_{nov},\, \mathcal{L}_{div},\, \mathcal{L}_{cyc},\, \mathcal{L}_{ce}; \\
\textbf{16:}\ \quad\quad\quad \mathcal{L}_G = -\lambda_d (\mathcal{L}_{nov} + \mathcal{L}_{div}) + \lambda_c \mathcal{L}_{cyc} + \lambda_{ce} \mathcal{L}_{ce}; \\
\textbf{17:}\ \quad\quad\quad g \leftarrow g + \tfrac{1}{|\mathcal{C}_{\mathrm{valid}}|} \nabla_\theta \mathcal{L}_G; \\
\textbf{18:}\ \quad\quad \textbf{end for} \\
\textbf{19:}\ \quad\quad \text{Clip } g;\ \text{update } G_\theta:\ \theta \leftarrow \theta - \eta_G\, g; \\
\textbf{20:}\ \quad \textbf{end for} \\
\textbf{21:}\ \textbf{end for} \\
\textbf{22:}\ \textbf{return}\ G_\theta\ \text{(and } F_\psi\ \text{for checkpoint selection)} \\
\hline
\end{array}
$$

### 2.4 Hyperparameters

Table 2.1 summarizes the key hyperparameters used for training the generator on ActivityNet.

**Table 2.1. Training hyperparameters for the VidInc-Synthetic generator.**

| Parameter | Value |
|---|---|
| Generator learning rate $\eta_G$ | $10^{-4}$ |
| Auxiliary classifier learning rate $\eta_F$ | $5\times10^{-4}$ |
| Weight decay (both) | $10^{-5}$ |
| Batch size $B$ (both stages) | 32 |
| $k_{\mathrm{shot}}$ per class | 32 |
| Epochs $E$ | 25 |
| Iterations per epoch $I$ | 10 |
| Pre-training epochs for $\phi$ | 15 |
| $K_n$ (number of novel domains) | 3 |
| Latent dimension $H$ | 768 |
| Transformer layers | 2 |
| Transformer attention heads | 4 |
| $\lambda_d$ | 0.5 |
| $\lambda_c$ | 0.01 |
| $\lambda_{ce}$ | 0.1 |
| Sinkhorn regularization $\varepsilon$ | 0.5 |
| Mixing weight $\alpha$ | 0.5 |
| Optimizer | Adam |

### 2.5 Model Configuration

Table 2.2 lists the exact specifications of all networks involved.

**Table 2.2. Architectural details of the temporal generator and auxiliary classifiers.**

| Module | Layer / Component | Specification |
|---|---|---|
| $G_\theta$ | Input | $\mathbf{X}\in\mathbb{R}^{25\times 768}$, L2-normalized |
| $G_\theta$ | Frame Projection | Linear(768, 768) |
| $G_\theta$ | Domain Embedding $\mathbf{E}$ | Embedding(4, 768), orthogonal init, scaled by $\sqrt{768}$ |
| $G_\theta$ | Transformer Encoder | 2 layers, $d_{\mathrm{model}}=768$, 4 heads, Pre-LN, no dropout |
| $G_\theta$ | Output Projection $\mathbf{W}_{\mathrm{out}}$ | Linear(768, 768), zero-initialized |
| $F_\psi$ and $\phi$ (identical) | MLP | Linear(768, 128), no bias |
| $F_\psi$ and $\phi$ (identical) | MLP | LayerNorm(128, no bias) |
| $F_\psi$ and $\phi$ (identical) | MLP | GELU activation |
| $F_\psi$ and $\phi$ (identical) | MLP | Linear(128, 200), no bias |

### 2.6 Data Pre-processing

All video features are extracted with CLIP ViT-L/14, taking $T=25$ uniformly sampled frames per video. Each frame feature is L2-normalized before being fed to the generator. During training, the full dataset is used to train $F_\psi$, while $G_\theta$ is trained with $k_{\mathrm{shot}}$-shot batches drawn per class to maintain balanced gradient signals across the $200$ classes.

---

## 3. Experimental Setups

### Evaluation Metrics

Following Lopez et al. (2017), we adopt **Average Accuracy**:

$$
\mathrm{ACC} = \frac{1}{T}\sum_{i=1}^{T} A_{T,i}
$$

and **Backward Transfer**:

$$
\mathrm{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1} (A_{T,i} - A_{i,i}),
$$

where $T$ is the total number of sequential tasks. BWT serves as a critical proxy for quantifying catastrophic forgetting: negative values indicate performance decay on earlier tasks. Additionally, **Maximum Mean Discrepancy (MMD)** with a Gaussian RBF kernel quantifies the magnitude of distribution shifts between domains.

### Representative Baselines

We evaluate three paradigms:

1. **Regularization-based:** EWC (Kirkpatrick et al., 2017), SI (Zenke et al., 2017).
2. **Replay-based:** ER (2022 Remember), GDumb (Prabhu et al., 2020), DER++ (Buzzega et al., 2020).
3. **Bounds:** `Fine-tuning`, which trains sequentially on each task without access to previous data and without any forgetting mitigation; `Joint Training`, which trains over the entire data stream simultaneously.

### Implementation Details

CLIP ViT-L/14 (Radford et al., 2021) extracts 768-dimensional features from 25 uniformly sampled frames per video, following the observation that self-supervised pre-trained representations provide a strong prior for incremental learning (Fini et al., 2022). The baseline architecture consists of a 2-layer Transformer encoder (hidden dimension 768, 8 attention heads) followed by a linear classification head. Optimization uses Adam (Kingma & Ba, 2014) ($lr=10^{-4}$, batch size 32), and replay methods employ a fixed buffer of 2,000 samples.

### Protocol Specifics per Benchmark

- **VidInc-Class:** Models are trained on sequential tasks $\mathcal{T}_1, \cdots, \mathcal{T}_{10}$ using a fixed stochastic seed (42) for task partitioning. Final performance is quantified by Average Accuracy (ACC) and Backward Transfer (BWT) after the full stream integration.
- **VidInc-CrossDomain:** Models encounter three domain-specific tasks sequentially, each covering the full 43-category label space. To account for sensitivity to domain ordering, we report the mean and standard deviation across all six possible domain permutations.
- **VidInc-Synthetic:** Models follow a fixed curriculum (Source $\rightarrow$ syn1 $\rightarrow$ syn2 $\rightarrow$ syn3) consisting of four tasks within a 200-category space. ACC and BWT are reported to measure adaptation capacity and the magnitude of cross-domain interference.

Incremental tasks are optimized for 15–20 epochs to achieve peak performance, whereas the `Joint Training` baseline utilizes 100 epochs to guarantee full convergence on the aggregated corpus.

---

## 4. Per-Order Breakdown for VidInc-CrossDomain

Table 4.1 and Table 4.2 provide the per-order ACC and BWT for all methods. A: ActivityNet, F: FCVID, U: UCF101.

**Table 4.1. Per-Order ACC (%) on VidInc-CrossDomain.**

| Method | A→U→F | A→F→U | U→A→F | U→F→A | F→A→U | F→U→A | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuning | 42.86 | 56.61 | 33.43 | 49.16 | 54.20 | 65.09 | 50.22 ± 11.09 |
| EWC | 58.44 | 58.09 | 47.63 | 57.41 | 65.13 | 66.22 | 58.82 ± 6.67 |
| SI | 28.84 | 13.59 | 24.43 | 6.52 | 33.14 | 29.86 | 22.73 ± 10.45 |
| ER | 67.30 | 78.09 | 66.92 | 73.82 | 86.13 | 84.92 | 76.20 ± 8.36 |
| GDumb | 73.28 | 77.46 | 77.60 | 75.67 | 76.62 | 74.69 | 75.89 ± 1.69 |
| DER++ | **84.13** | **84.09** | **86.35** | **90.58** | **91.35** | **91.36** | **87.98 ± 3.52** |

**Table 4.2. Per-Order BWT (%) on VidInc-CrossDomain.**

| Method | A→U→F | A→F→U | U→A→F | U→F→A | F→A→U | F→U→A | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuning | -72.04 | -54.08 | -86.55 | -64.07 | -59.07 | -42.98 | -63.12 ± 16.42 |
| EWC | -46.93 | -46.67 | -59.30 | -47.56 | -38.75 | -35.79 | -45.83 ± 8.34 |
| SI | -84.16 | -85.64 | -87.17 | -93.16 | -87.74 | -91.80 | -88.28 ± 3.46 |
| ER | -36.25 | -19.00 | -33.32 | -27.41 | -10.09 | -12.43 | -23.08 ± 11.83 |
| GDumb | **10.91** | **13.28** | **4.41** | **1.44** | **33.59** | **10.56** | **9.45 ± 12.03** |
| DER++ | -7.38 | -9.08 | -7.74 | -2.35 | -1.69 | -1.12 | -3.45 ± 3.71 |
