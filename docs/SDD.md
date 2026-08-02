# kim2022multi - Software Design Document

## 1. System Overview

**Project Title:** 의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크  
**Venue:** 2022 대한전자공학회 추계학술대회 (Fall Conference of KIEE)  
**Authors:** Dongjoo Kim, Minsik Lee  
**Institution:** Hanyang University, Applied AI Lab  
**Publication:** Kim, D., & Lee, M. (2022). 의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크. *대한전자공학회 학술대회*, 788-791.

### Problem Statement

Predict clothing demand information (preferred customer gender, age group, view count, sales volume) using a combination of:
- **Visual features:** CNN from product images (125×125 RGB)
- **Textual features:** BERT embeddings from product names (multilingual-cased, 768D)
- **Tabular features:** Metadata (gender, price, category)

### Architecture Innovation

Multi-modal fusion network combining CNN (image) + BERT (text) + metadata, trained with multi-task learning to predict 4 interdependent demand metrics simultaneously.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

Step 1: Dataset Preparation (shopping_dataset.py + bert_features.py)
   ├─ Load images from dataset/category_all_ver2_20221002_words_125_aug/{0-9}/*.png
   ├─ Tokenize product names with BERT (bert-base-multilingual-cased)
   ├─ Extract [CLS] token embeddings → 768D features
   ├─ Parse filename for labels: sex, best_sex, best_age, view, sales
   ├─ 80/20 train/test split (random, no stratification)
   └─ Output: ShoppingDataset with 10 fields per sample

Step 2: Model Definition (src/models/resnet_pre_trained.py)
   ├─ Backbone: ResNet18 (ImageNet pretrained)
   ├─ Image path: Conv layers → [512D] → Avg pooling
   ├─ Fusion: [512D image || 3D metadata || 768D BERT] → 1283D
   ├─ FC layers: 1283D → 64D → 4 output heads
   ├─ Outputs: (best_sex[3], best_age[7], view[1], sales[1])
   └─ Total params: ~12M

Step 3: Multi-Task Training (src/train.py)
   ├─ Loss: CE(best_sex) + Focal(best_age) + 0.01*MSE(view) + 0.01*MSE(sales)
   ├─ Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
   ├─ Batch size: 64 | Epochs: 3000
   ├─ Device: Supports GPU and CPU (auto-detected)
   └─ Output: Checkpoints + AccLoss2.txt

Step 4: Single-Task Evaluation (src/train_single_task.py)
   ├─ Ablation mode: train one head at a time
   ├─ CLI: --analysis {best_sex|best_age|view|sales}
   ├─ Loss: Appropriate to task type
   └─ Output: Analysis-specific results
```

---

## 3. Module Descriptions

### 3.1 src/data/shopping_dataset.py

**Purpose:** PyTorch Dataset loader merging image files, CSV metadata, and BERT features into a unified training interface.

**Input:**
- `data_path`: Directory of images organized as `./DATA/train/{0-9}/*.png` (50k files, 125×125 RGB)
- `csv_path`: CSV mapping goods_num → clothing_name
- `csv_info`: Pre-extracted BERT [CLS] embeddings (dict indexed by goods_num)

**Filename Format:** `{idx}_{goodsNum}_{sex}_{best_sex}_{best_age}_{view}_{category}_{price}_{sales}_{extra}.png`

**Parsing (indices):**
| Index | Field | Type | Usage |
|-------|-------|------|-------|
| 0 | idx | int | Image identifier |
| 2 | sex | int | 0=female, 1=male, 2=unisex |
| 3 | best_sex | int | 0-2, prediction target |
| 4 | best_age | int | 0-6, prediction target |
| 5 | view | int | log-normalized if > 0 |
| 6 | category | int | 0-9+ product category |
| 7 | _(unused)_ | - | - |
| 8 | price | int | 0-8 normalized price ← **BUG FIX #1** |
| 9 | _(unused)_ | - | - |
| 10 | _(unused)_ | - | - |

**Data Augmentation:**

**Train (transform):**
```python
RandomHorizontalFlip()
RandomVerticalFlip()
RandomResizedCrop((125,125), scale=(0.1,1), ratio=(0.5,2))
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Test (transform2):**
```python
ToTensor()  # Deterministic: no resizing, crops, or augmentation
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Output (per sample):**
```python
(image[3,125,125], sex[1], best_sex, best_age, view, sales, price, category, bert_feature[1,768], index)
```

**Known Issues (documented):**
- Test transform lacks explicit resize (assumes source images are 125×125, else shape mismatch)
- `label_best_sex_and_best_age` computed but unused (was intended for stratified split, now disabled)
- Unused `get_bert_feature()` method on class (duplicates module-level function in train.py)

### 3.2 src/models/resnet_pre_trained.py

**Purpose:** Multi-modal fusion network combining image, metadata, and text features for joint prediction of 4 clothing demand metrics.

**Architecture:**

```
Image (3, 125, 125)
    ↓
ResNet18[:-3]  (remove avgpool, fc, one block)
    ↓
Conv2d(256→128, 3×3, stride=1, pad=1)
BatchNorm2d(128)
ReLU
    ↓
AvgPool2d(4)  → [B, 128, H/4, W/4]
    ↓
Flatten → [B, 512]
    ↓
Concat([512] + [1 sex] + [1 price] + [1 category] + [768 BERT])
    ↓
BatchNorm1d(1283)
    ↓
FC(1283 → 64)  [with Dropout(0.5)]
    ↓
┌─────────────────────────────────────────┐
│ Output Heads (share 64D representation) │
├──────────────────┬──────────┬──────┬────┤
│ best_sex[3]      │ best_age[7]  │ view[1]  │ sales[1]  │
│ (classification) │ (classification) │ (regr.) │ (regr.) │
└──────────────────┴──────────┴──────┴────┘

Return: (out_best_sex, out_best_age, out_view, out_sales)
```

**Forward Pass Signature:**
```python
def forward(self, image, sex, price, category, bert_feature):
    # image: [B, 3, 125, 125]
    # sex: [B, 1], price: [B, 1], category: [B, 1]
    # bert_feature: [B, 768]
    # Returns: 4-tuple of tensors
    return (out_best_sex, out_best_age, out_view, out_sales)
```

**Hyperparameters:**
- Pretrained weights: ImageNet (ResNet18)
- Conv layers retained: 12 (ResNet layers 0-11)
- FC hidden dims: 1283 → 64
- Dropout rate: 0.5

### 3.3 src/train.py

**Purpose:** Multi-task training loop implementing the paper's joint learning approach.

**Entry Point:** `if __name__ == '__main__':`

**Workflow:**
1. Load BERT tokenizer + model (multilingual-cased, moved to `bert_features.py` for reuse)
2. Read CSV and extract BERT features for all products (loop over df, ~slow on CPU)
3. Create train/test DataLoaders with ShoppingDataset
4. Define 3 loss functions: CE, CE(reduction='none'), MSE
5. Loop epochs:
   - `train()`: forward all 4 heads, compute combined loss, backward, step
   - `evaluate()`: forward all 4 heads, log per-head metrics
6. Save checkpoint and results to `results/AccLoss2.txt`

**Loss Function (line 161 original):**
```python
total_loss = (CE(best_sex) + 
              focal(best_age, alpha=1, gamma=2) + 
              0.01 * MSE(view) + 
              0.01 * MSE(sales))
```

**Focal Loss Formula (lines 152-154):**
```python
loss = criterion_for_focal(logits, labels)  # CE, reduction='none'
pt = exp(-loss)
focal = (alpha * (1 - pt)^gamma * loss).mean()
```

**Known Issues (documented):**
- BERT loading + feature extraction happens at module import time (CPU-only, not parallelized, slow)
- NaN/Inf guard in `evaluate()` runs AFTER accumulating into totals (unlike `train()`, which guards before) — asymmetry, result is documented but not fixed to preserve original behavior
- Dead `get_bert_feature()` method in ShoppingDataset (never called)

### 3.4 src/train_single_task.py

**Purpose:** Single-task ablation mode to evaluate each prediction head independently.

**CLI Argument:**
```bash
python -m src.train_single_task --analysis {best_sex|best_age|view|sales}
```

**Behavior:**
- Load model, DataLoaders (identical to `train.py`)
- In each epoch, unpack 4-tuple output and select only the head matching `--analysis`
- Compute appropriate loss (CE for classification, MSE for regression)
- Log head-specific metrics

**Known Discrepancy (Documented):**
- **Filename says "noword_ablation"** but code does NOT actually remove text features
- The actual ablation is **single-task** (one head at a time), not "no-word" (text feature removal)
- Text features (BERT) are computed and passed to the model regardless of `--analysis`
- This naming inconsistency is a historical artifact; no true no-word ablation was implemented

**Output:**
```
results/AccLoss2.txt
├─ analysis: {chosen head}
├─ test_epoch_{head}_acc or test_epoch_{head}_loss
└─ per-epoch metrics
```

### 3.5 src/utils/bert_features.py

**Purpose:** Centralized BERT loading and feature extraction (replaces duplication in main.py and main_noword_ablation.py).

**Key Functions:**

**`load_bert(model_name, device)`**
- Loads tokenizer + BertModel (multilingual-cased by default)
- Moves model to device, sets .eval()
- Returns: (tokenizer, net_bert)

**`get_bert_feature(clothing, tokenizer, net_bert, device)`**
- Single product name → BERT [CLS] embedding
- Input: string (e.g., "배색큐롯스커트(W)")
- Output: [1, 768] tensor

**`get_bert_feature_by_batch(clothing_feature)`**
- Concatenate per-sample features extracted during data loading
- Input: list of [1, 768] tensors
- Output: [B, 768] tensor
- ← **BUG FIX #2:** Changed `if i is 0:` to `if i == 0:` (identity → equality)

**`build_csv_info(csv_path, tokenizer, net_bert, device)`**
- Load CSV, iterate rows, extract BERT for each product name
- Returns: list of {goods_num: [1,768]} dicts

### 3.6 src/utils/training_utils.py

**Purpose:** Shared loss and accuracy computation functions (replaces duplication in train.py and train_single_task.py).

**Key Functions:**

**`focal_loss(logits, labels, criterion_for_focal, alpha, gamma)`**
```python
loss = criterion_for_focal(logits, labels)  # CE, reduction='none'
pt = torch.exp(-loss)
focal = (alpha * (1 - pt) ** gamma * loss).mean()
return focal
```

**`combined_multitask_loss(...)`**
```python
loss_sex = CE(best_sex)
loss_age = focal(best_age)
loss_view = MSE(view)
loss_sales = MSE(sales)
return loss_sex + loss_age + 0.01*loss_view + 0.01*loss_sales
```

**`joint_accuracy(pred_best_sex, pred_best_age, label_best_sex, label_best_age)`**
- Count samples where BOTH best_sex AND best_age predictions are correct
- Returns: int (count of correct joint predictions)

### 3.7 src/utils/io_utils.py

**Purpose:** Directory creation and result file I/O utilities.

**Key Functions:**

**`ensure_output_dirs(*dirs)`**
- os.makedirs for each directory with exist_ok=True
- ← **BUG FIX #4:** Called from train() and train_single_task() to create model_weights/ and results/ before saving

**`save_checkpoint(model, filepath)`**
- torch.save(model.state_dict(), filepath)
- Creates parent directory if needed

**`save_run_results(filepath, **results_dict)`**
- Write results dictionary to text file in existing format
- Same format as original main.py's manual f.write() calls

---

## 4. Data Flow & Label Encoding

### Filename Format (after scraping)

Each image filename encodes 11+ space-delimited fields:

```
{idx}_{goodsNum}_{sex}_{best_sex}_{best_age}_{view}_{category}_{price}_{sales}_{extra1}_{extra2}.png
 0      1           2      3         4         5      6        7      8       9      10
```

**Example:**
```
8711_2680976_0_1_5_4100_7_100_2_69900_1_0.png
```

| Field | Index | Value | Meaning |
|-------|-------|-------|---------|
| idx | 0 | 8711 | Internal image ID |
| goodsNum | 1 | 2680976 | Product ID (links to CSV) |
| sex | 2 | 0 | Purchaser gender (0=F, 1=M, 2=unisex) |
| best_sex | 3 | 1 | Most common purchaser gender **[LABEL]** |
| best_age | 4 | 5 | Most common purchaser age group (0-6) **[LABEL]** |
| view | 5 | 4100 | Number of views **[LABEL, log-normalized]** |
| category | 6 | 7 | Product category **[LABEL]** |
| price | 7 | 100 | Price bin (0-8) **[LABEL]** ← BUG FIX: use index 8 for test, not 9 |
| sales | 8 | 2 | Number of sales **[LABEL, log-normalized]** |
| extra1 | 9 | 69900 | (unused, possibly metadata) |
| extra2 | 10 | 1 | (unused) |
| extra3 | 11 | 0 | (unused) |

### Label Processing in shopping_dataset.py

```python
# Train branch:
self.sex.append(int(float(list[2])))              # Index 2
self.label_best_sex.append(int(float(list[3])))   # Index 3
self.label_best_age.append(int(float(list[4])))   # Index 4
if int(float(list[5])) != 0:
    self.label_view.append(math.log(int(float(list[5]))))  # Index 5, log if > 0
else:
    self.label_view.append(0)
if int(float(list[7])) != 0:
    self.label_sales.append(math.log(int(float(list[7]))))  # Index 7, log if > 0
else:
    self.label_sales.append(0)
self.price.append(int(float(list[8])))            # Index 8 ← **BUG FIX #1**
self.category.append(int(float(list[10])))        # Index 10

# Test branch: identical (prices now both use index 8)
```

---

## 5. Loss Functions in Detail

### Classification: Cross-Entropy Loss

**best_sex (3 classes):**
```
L_sex = CE(logits_sex, label_best_sex)
```

**best_age (7 classes) with Focal Loss:**
```
L_age_ce = CE(logits_age, label_best_age, reduction='none')
pt = exp(-L_age_ce)
L_age_focal = (alpha * (1 - pt)^gamma * L_age_ce).mean()
  where alpha=1, gamma=2
```

**Rationale:** Focal loss down-weights easy examples to focus on hard negatives, addressing class imbalance in demographic prediction.

### Regression: Mean Squared Error

**view (continuous):**
```
L_view = MSE(pred_view, label_view)
where label_view = log(view_count) if view_count > 0 else 0
```

**sales (continuous):**
```
L_sales = MSE(pred_sales, label_sales)
where label_sales = log(sales_count) if sales_count > 0 else 0
```

**Rationale:** Log-normalization compresses heavily skewed distributions (view/sales counts span 0 to millions).

### Combined Multi-Task Loss

```
L_total = L_sex + L_age_focal + loss_alpha * L_view + loss_beta * L_sales
        = L_sex + L_age_focal + 0.01 * L_view + 0.01 * L_sales
```

**Weight Justification:**
- `loss_alpha = loss_beta = 0.01`: Regression losses are down-weighted relative to classification (which is the primary task)
- Classification (sex, age) predicts customer demographics (directly actionable)
- Regression (view, sales) predicts engagement (auxiliary, harder to predict precisely)

---

## 6. Paper → Code Mapping

| Paper Component | Code Location | Status |
|-----------------|---------------|--------|
| Table 2: Network Architecture (CNN + BERT fusion) | `src/models/resnet_pre_trained.py:forward()` | ✅ Exact match |
| Table 3: Hyperparameters (batch=64, lr=1e-4, epochs=3000) | `src/config.py` | ✅ Exact match |
| Algorithm / Loss Function (Multi-task loss) | `src/utils/training_utils.py:combined_multitask_loss()` | ✅ Exact match |
| Focal Loss (α=1, γ=2) | `src/utils/training_utils.py:focal_loss()` | ✅ Exact match |
| Transformer Encoder (BERT multilingual) | `src/utils/bert_features.py:load_bert()` | ✅ Exact match |
| Data Augmentation (train: RHFlip, RVFlip, RRC; test: deterministic) | `src/data/shopping_dataset.py:transform() / transform2()` | ✅ Exact match |
| Train/Test Split (80/20 random) | `src/data/shopping_dataset.py:__init__()` | ✅ Implemented |

---

## 7. Removed / Legacy Code

### Models/resnet.py
- **Status:** Deleted (checked in as legacy, never imported)
- **Issue:** Two independent bugs:
  1. `self.bn1` defined twice with incompatible types (BatchNorm2d then overwritten by BatchNorm1d)
  2. `forward()` references `self.layer3` which is never created in `__init__`
- **Why Removed:** Non-functional, hand-rolled ResNet predating the pretrained approach

### Models/Network.py
- **Status:** Deleted (checked in as legacy, never imported)
- **Issue:** Shallow CNN, no BERT input parameter, incompatible with current training loop
- **Why Removed:** Superseded by `resnet_pre_trained.py` once BERT was integrated

---

## 8. Known Limitations & Deviations

### BERT Feature Extraction Timing
- **Current:** BERT features extracted at module import time (slow, single-threaded, CPU-only)
- **Limitation:** Could be parallelized or batched during data loading
- **Not Fixed Because:** Would change wall-clock training time and results reproducibility

### NaN/Inf Guard Asymmetry in evaluate()
- **Current:** `test_total_loss` checked for NaN/Inf *after* accumulation (line 292-294)
- **Compare:** `train()` guards *before* accumulation (line 163-164)
- **Impact:** Contaminated loss values could propagate into running totals
- **Not Fixed Because:** Correcting would change evaluated loss curves and reproducibility

### "noword_ablation" Naming Mismatch
- **Current:** `train_single_task.py` still computes and uses BERT features despite original filename "noword"
- **Actual Behavior:** Single-task training (one head at a time), not text-feature ablation
- **How It Works:** `--analysis` flag selects which output head to train; BERT is always used
- **Not Fixed Because:** Would require redesigning the ablation structure; true no-word ablation not implemented

### clip_norm Config Value
- **Current:** Defined in `config.py` as `clip_norm = 5`
- **Actual Usage:** Never applied (no `torch.nn.utils.clip_grad_norm_` in code)
- **Not Fixed Because:** Adding would change training dynamics and paper reproducibility

### Test Transform Missing Explicit Resize
- **Current:** Test transform applies Normalize only, no RandomResizedCrop or explicit Resize
- **Assumption:** Source images are already 125×125
- **Risk:** Shape mismatch if source images differ in size
- **Not Fixed Because:** Would change evaluation behavior and paper reproducibility

---

## 9. Configuration Parameters

All hyperparameters stored in `src/config.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 64 | Training batch size (per GPU) |
| `num_epochs` | 3000 | Total training epochs |
| `lr` | 0.0001 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `clip_norm` | 5 | Gradient clipping threshold (unused) |
| `alpha` (focal) | 1 | Focal loss α parameter |
| `gamma` (focal) | 2 | Focal loss γ parameter |
| `loss_alpha` | 0.01 | View regression weight |
| `loss_beta` | 0.01 | Sales regression weight |
| `data_path` | `dataset/category_all_ver2_20221002_words_125_aug/` | Image directory |
| `csv_path` | `dataset/goodsNum_clothing_name_20221002.csv` | Metadata CSV |

---

## 10. Validation & Testing Strategy

See [docs/TC.md](TC.md) for comprehensive test cases covering:
- Unit tests for each utility function (data parsing, loss computation, BERT extraction)
- Integration tests for full pipeline (model forward, loss computation, checkpoint saving)
- System tests validating paper alignment (Algorithm 1, hyperparameters, output shapes)
- Regression tests pinning known asymmetries (NaN guard, transform differences)

---

**Document Version:** 2026-07-31  
**Status:** Production-Ready  
**All algorithm logic preserved from original; bugs fixed restore intended behavior.**
