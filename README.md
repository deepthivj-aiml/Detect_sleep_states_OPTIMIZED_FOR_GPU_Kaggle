# Detect Sleep States — Approach, Evaluation, and GPU Notes

This README documents the approach used in `detect-sleep-states.ipynb`, explains how the
F1 scores are computed and what they mean, and clarifies which GPU optimization
techniques are (and are **not**) actually present in this particular run.

---

## 1. Problem

Given wrist-accelerometer readings (`anglez`, `enmo`) sampled every 5 seconds for many
people over many nights, predict the exact timestep of each **sleep onset** and
**wakeup** event.

---

## 2. Approach / Pipeline

### 2.1 Data cleaning
- `train_series.parquet` (raw sensor readings), `train_events.csv` (onset/wakeup
  labels), and `test_series.parquet` are loaded with Polars.
- Sensor columns are downcast from `float64` → `float32` immediately after load, to cut
  memory roughly in half for every downstream step.
- A small number of nights have an onset **or** a wakeup recorded but not both
  (`faulty_pairs`). These incomplete records are nulled out rather than partially used,
  since a one-sided event can't be turned into a valid "asleep" span.

### 2.2 Label construction
Each row needs a 0/1 `asleep` label, but the raw data only marks the *boundary* rows
(the exact onset/wakeup step) — not every row in between. The label is built by:
1. Marking onset rows `+1`, wakeup rows `-1`, everything else `0`.
2. Taking a running (`cumsum`) total per person.
3. Any row where that running total is `> 0` is between an onset and a wakeup →
   labeled `asleep = 1`.

This turns two sparse event markers per night into a dense per-row label the model can
actually be trained on.

### 2.3 Feature engineering (`create_features`)
A single point-in-time reading isn't predictive on its own — sleep is a pattern over
time. For 8 window sizes (1, 5, 15, 30, 60, 120, 240, 480 minutes), rolling `min`,
`max`, `mean`, and `std` are computed for:
- `anglez`, `enmo` (raw values)
- `anglez_diff`, `enmo_diff` (how much each changes between consecutive readings)

This produces the features the model actually trains on — a summary of "how still or
active was this person over the last X minutes," at multiple time scales.

### 2.4 Algorithm: incremental / batched XGBoost
**Why XGBoost:** gradient-boosted trees handle tabular, mixed-scale numeric features
well, train fast, and don't require the extensive scaling/normalization deep learning
approaches would need for this kind of engineered feature set.

**Why batched/incremental training, not `model.fit()` on the whole dataset:**
The full feature-engineered dataset (240+ rolling-window columns × millions of rows)
doesn't comfortably fit in memory at once. The data is split into ~1M-row batches; for
each batch:
1. Rolling-window features are computed.
2. A small `dtrain`/`dvalid` split (90/10) is carved out for early-stopping.
3. `xgb.train(..., xgb_model=booster)` **continues** training the same booster from the
   previous batch, instead of restarting — this is what makes it "incremental" rather
   than just training many unrelated models.
4. `early_stopping_rounds=10` halts a batch's boosting early if validation logloss
   isn't improving, to avoid a bad batch degrading the shared model.

**Class imbalance:** `scale_pos_weight` is computed from the actual asleep/awake row
counts and passed into `params`, so the model doesn't just learn to predict the
majority class.

### 2.5 Holdout split
`train_test_split` is applied at the **`series_id` (person) level**, not the row
level — 10% of *people* are held out entirely. This matters because a person's own
sensor stream is highly self-correlated; a row-level split would leak information about
a specific person's rhythm into "validation," inflating scores artificially.

### 2.6 Prediction → events
The trained booster outputs a per-row asleep probability. `predict()` converts this
back into discrete onset/wakeup events by:
1. Thresholding at 0.5 to get a 0/1 prediction per row.
2. Finding where the prediction *changes* (0→1 = onset, 1→0 = wakeup).
3. Discarding segments shorter than 30 minutes (noise filtering).
4. Merging segments separated by less than 2 hours (treats brief wake-ups mid-sleep as
   one continuous sleep period).
5. Keeping the single longest segment per night, scored by the model's mean confidence
   over that segment.

---

## 3. F1 Score — two different metrics, computed two different ways

The notebook actually reports **two distinct kinds of accuracy metric**, and they tell
you very different things. It's important not to conflate them.

### 3.1 Row-level accuracy
```
accuracy = (holdout_preds == y_ho).mean()   # → 0.9328
```
This checks, for every single 5-second row in the holdout set, whether the predicted
asleep/awake label matches the true label. **Result: 93.28%.**

This number looks strong but is somewhat misleading on its own: most rows in any given
night are *unambiguously* asleep or awake (the middle of a nap, the middle of a work
day) — only the rows right around a transition are hard. Since transitions are a small
fraction of total rows, a model can score very high on row-level accuracy while still
being fairly imprecise about *exactly which row* the transition happens on.

### 3.2 Event-level F1 (competition-style, tolerance-matched)
This is the metric that actually matters for the task — it checks whether the
*predicted onset/wakeup timestamps* land close enough to the *true* ones, not whether
every row in between was labeled correctly.

**Algorithm used** (for each event type — `onset` and `wakeup` — separately):
1. Define a tolerance window: `TOLERANCE = 12 * 30` steps = 360 steps = **30 minutes**
   (steps are 5 seconds apart, so 12 steps/minute).
2. For each **predicted** event, look for **unmatched** true events from the *same
   person* within ±30 minutes.
3. If one or more candidates exist, greedily match the predicted event to the
   **closest** unmatched true event, and mark both as used (so no true event can be
   matched twice).
4. After all predictions are processed:
   - **TP** (true positive) = number of predictions successfully matched to a true event
   - **FP** (false positive) = predictions with no matching true event within tolerance
   - **FN** (false negative) = true events that were never matched by any prediction
5. Standard precision/recall/F1 from those counts:
   ```
   precision = TP / (TP + FP)
   recall    = TP / (TP + FN)
   f1        = 2 · precision · recall / (precision + recall)
   ```
6. **Macro F1** = the unweighted average of the onset F1 and wakeup F1 — this is the
   single headline number.

**Why greedy nearest-match with a tolerance window, instead of exact-step matching:**
sleep timing predictions that are "close enough" (within 30 minutes) are still useful;
requiring an exact-step match would make the metric far too strict and uninformative
for comparing model variants. The tolerance and greedy-nearest-match approach mirrors
how this type of Kaggle competition (Child Mind Institute — Detect Sleep States)
actually scores submissions.

### 3.3 Actual results from this run

| Event  | TP  | FP  | FN  | Precision | Recall | F1     |
|--------|-----|-----|-----|-----------|--------|--------|
| onset  | 87  | 410 | 358 | 0.1751    | 0.1955 | 0.1847 |
| wakeup | 129 | 368 | 316 | 0.2596    | 0.2899 | 0.2739 |

**Macro F1 (onset + wakeup): 0.2293**

**Why row-level accuracy (0.9328) and event-level F1 (0.2293) look so different:**
these measure fundamentally different things. Getting 93% of individual rows right is
easy when most rows aren't near a transition; correctly pinpointing the *specific*
minute someone fell asleep or woke up — which is what the event-level F1 actually
tests — is a much harder problem, and the low F1 here indicates the model's transition
timing is still quite noisy relative to the 30-minute tolerance window, even though its
general asleep/awake classification is fairly accurate.

---

## 4. GPU Optimization — status in *this* notebook

**Important:** the notebook this README describes (the run with the results above) is
the **CPU-only** version. It does **not** use a GPU. Specifically:
```python
params = {
    'tree_method': 'hist',   # CPU histogram-based tree building
    'nthread': -1,           # uses all CPU cores, not a GPU device
    ...
}
dtrain = xgb_lib.DMatrix(...)   # plain CPU DMatrix
```
There is no `device='cuda'`, no `cudf`, and no `QuantileDMatrix` in this run. The
93-minute-ish training/evaluation time (≈47 minutes across the two training+holdout
loops) reflects CPU training.

### GPU techniques explored separately (not part of this run)
A GPU-accelerated variant of this pipeline was designed and iterated on elsewhere in
this project, using:

| Technique | What it does | Why it's used |
|---|---|---|
| `device='cuda'` + `tree_method='hist'` | Runs XGBoost's histogram-based tree building on GPU | Tree building is the dominant training cost; GPU histogram construction is typically a large speedup over CPU for this workload |
| `cudf` DataFrames | Moves feature data into GPU-native columnar arrays | Avoids CPU↔GPU transfer overhead during training — data stays on-device |
| `QuantileDMatrix` | Stores training data as compressed, pre-binned histograms instead of raw values | Much lower GPU memory footprint than a plain `DMatrix`, which is essential given GPU memory is far more constrained than system RAM |
| `GroupKFold` (5-fold, person-level) | Splits by `series_id` across 5 folds instead of one holdout | More reliable performance estimate (mean ± variance across folds) than a single train/val split; also unlocks ensembling |
| RMM pool reinitialization per fold (`init_rmm_pool()`) | Explicitly resets the GPU memory pool allocator between folds | Pool allocators don't return memory to the OS on `free()`; periodic reinitialization avoids cross-fold memory fragmentation and OOM errors |
| Batched, shuffled-series training | Processes ~400K-row chunks at a time, with people shuffled across batches | Keeps peak GPU memory bounded, while avoiding batches dominated by a narrow slice of people |
| 5-model ensembling (`predict_ensemble`) | Averages predicted probabilities across all 5 fold models at inference time | Reduces prediction variance — different folds make different, partially uncorrelated errors, and averaging tends to cancel some of them out |

None of this GPU/ensemble machinery is reflected in the F1/accuracy numbers reported in
Section 3 above — those numbers come purely from the single-holdout, CPU-trained model.
If/when the GPU + 5-fold ensemble pipeline is run end-to-end, it should be evaluated
with the same event-level F1 methodology (Section 3.2) to get a directly comparable
number.

---

## 5. Known limitations worth noting

- **Event-level F1 is currently low (0.2293).** The rolling-window feature set and 0.5
  probability threshold are reasonable starting points but likely leave meaningful
  accuracy on the table — a threshold sweep (`precision_recall_curve`) or richer
  features (e.g. the full 15-window set, rather than the trimmed 8) are natural next
  steps.
- **Post-processing keeps only one segment per night** — if a person has genuinely
  fragmented sleep across a night, only the single longest segment is reported, which
  could suppress legitimate secondary events.
- **The 90/10 in-batch validation split (Section 2.4) is not the same as the holdout
  split (Section 2.5)** — the former is used only for early-stopping signal during
  training; the latter is the actual, honest evaluation set used for the F1 numbers
  above.
