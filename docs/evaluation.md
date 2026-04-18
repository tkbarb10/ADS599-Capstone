# Model Evaluation Results

## Dataset Overview

All models draw from the same MIMIC-IV ED cohort: adult ED visits with a terminal outcome of discharge or ICU transfer. The class imbalance is consistent across all splits (~7.8-8.0% ICU transfer rate).

---

## Training Data Characteristics by Model

### Traditional ML (Logistic Regression, Random Forest, XGBoost)

**Source:** `full_patient_state` -- aggregated to one row per stay, using the first 60 minutes of each visit. Vital values are taken at triage time (`first` within window); binary event flags use worst-case aggregation (`max` within window).

**Split:** 80/10/10 stratified by `subject_id` (not `ed_stay_id`) to prevent leakage across repeat visits by the same patient.

| Split | Stays | Discharge | ICU Transfer | ICU Rate |
|---|---|---|---|---|
| Train (est.) | ~314,600 | ~289,700 | ~24,900 | ~7.9% |
| Validation | 39,222 | 36,192 | 3,030 | 7.7% |
| Test | 39,325 | 36,234 | 3,091 | 7.9% |

**Feature columns:**

| Model | Feature Cols | Chief Complaint Encoding |
|---|---|---|
| Logistic Regression | 242 | OHE -- 21 clinical categories |
| Random Forest | 242 | OHE -- 21 clinical categories |
| XGBoost | 271 | TF-IDF -- top 50 unigrams/bigrams (replaces OHE cc_ columns) |

The 224 shared features include: current vitals (6), derived MAP, lab state OHE (57), microbiology OHE (84), dispensed medication flags (21), medrecon flags (19), ECG/radiology OHE (6), action flags (5), static patient features (anchor_age, acuity, gender, height, weight, arrival transport OHE), and missingness indicators. XGBoost drops the 21 OHE chief complaint columns and substitutes 50 TF-IDF terms, netting 271 total.

---

### LSTM

**Source:** `full_patient_state` -- full event-driven time series per stay, variable-length sequences padded to 100 events.

**Split:** Same 80/10/10 stratified by `subject_id` as traditional ML. Minor difference in test count vs. traditional ML (39,520 vs. 39,325) comes from sequence filtering: stays with fewer than 2 events are retained for the one-row traditional ML aggregation but excluded from LSTM sequences.

| Split | Stays | Discharge | ICU Transfer | ICU Rate |
|---|---|---|---|---|
| Train (est.) | ~316,160 | ~290,900 | ~25,260 | ~7.9% |
| Validation (est.) | ~39,520 | ~36,400 | ~3,120 | ~7.9% |
| Test | 39,520 | 36,473 | 3,047 | 7.7% |

**Feature columns:** 236 (same feature space as traditional ML minus the chief complaint OHE columns; chief complaint is not included in the sequential state).

---

### RL Agent (sbs_predictions)

**Source:** `sbs_predictions` -- same feature space as `full_patient_state` (236 features) plus `p_icu` (LSTM step-by-step ICU probability) and `delta_p_icu` (change in p_icu from prior event). The step-by-step LSTM inference uses a different split from the standard LSTM training: `train_size = 0.6`, holding out 40% of all stays for step-by-step prediction so the RL agent trains on held-out data only.

The RL agent then splits `sbs_predictions` 80/20 by `ed_stay_id`.

| Split | Stays | Rows (events) | Discharge | ICU Transfer | ICU Rate |
|---|---|---|---|---|---|
| sbs_predictions total | 158,309 | 2,770,657 | ~146,040 | ~12,269 | ~7.8% |
| RL Train | 127,042 | ~2,216,525 | 117,274 | 9,768 | 7.7% |
| RL Test | 31,267 | ~554,132 | 28,780 | 2,487 | 7.95% |

**State features:** 236 (includes `p_icu` and `delta_p_icu` within the 236 input dimensions).

---

## Raw Metrics: Traditional ML + LSTM

Metrics are computed on the test split for the positive class (ICU transfer, label = 1) using binary classification metrics (sklearn defaults). Accuracy is overall. ROC-AUC and PR-AUC are probability-based.

| Model | Accuracy | Precision | Recall | F1 (ICU) | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| LSTM | 0.9884 | 0.8994 | 0.9560 | 0.9268 | 0.9942 | 0.9770 | 0.0113 |
| XGBoost | 0.8405 | 0.3037 | 0.7962 | 0.4397 | 0.9056 | 0.6067 | 0.1125 |
| Random Forest | 0.9048 | 0.4282 | 0.6283 | 0.5093 | 0.8949 | 0.5416 | 0.0901 |
| Logistic Regression | 0.8155 | 0.2696 | 0.7881 | 0.4017 | 0.8907 | 0.5253 | 0.1318 |

**Notes:**
- Precision, Recall, and F1 are for the ICU transfer class (positive class). Accuracy is influenced heavily by the ~92% discharge majority.
- PR-AUC is the primary ranking metric given class imbalance. LSTM (0.977) is dramatically stronger than all traditional ML models (0.52-0.61).
- LSTM Brier Score (0.011) indicates well-calibrated probabilities. Traditional ML models have much higher Brier Scores (0.09-0.13).
- XGBoost achieves the highest ROC-AUC (0.9056) among traditional ML, but its PR-AUC (0.607) still trails LSTM by a wide margin.

---

## LSTM Training Loss Curves

| Epoch | Train Loss | Eval Loss |
|---|---|---|
| 1 | 0.1065 | 0.0727 |
| 2 | 0.0932 | 0.0700 |
| 3 | 0.0902 | 0.0750 |
| 4 | 0.0904 | 0.0781 |
| 5 | 0.0927 | 0.0709 |

Best eval loss: 0.0700 at epoch 2. The model ran all 5 configured epochs; early stopping patience=3 was not triggered (eval loss at epoch 5 returned close to best). The tight train/eval gap indicates good generalization with minimal overfitting.

---

## Raw Metrics: RL Agent

### Training Loss (3 epochs)

| Epoch | Train Loss | Q-Loss | CQL | Eval Loss |
|---|---|---|---|---|
| 1 | 0.8175 | 0.7974 | 0.2014 | 0.4010 |
| 2 | 0.4236 | 0.4095 | 0.1407 | 0.4723 |
| 3 | 0.3898 | 0.3766 | 0.1320 | 0.4733 |

Train loss converges after epoch 2. Eval loss increases after epoch 1 and plateaus, indicating the agent overfits to the training distribution after sufficient training -- expected behavior in offline RL since the agent cannot explore to reduce distributional shift.

### Terminal Row Metrics

Evaluated on the terminal event of each test stay (the row where the provider made their actual disposition decision). There are 31,267 terminal rows -- one per test stay.

| Metric | Value |
|---|---|
| Terminal rows evaluated | 31,267 |
| Agent predicted wait | 18,576 (59.4%) |
| Agent predicted discharge or transfer | 12,691 (40.6%) |
| F1 -- discharge (acted rows only) | 0.9990 |
| F1 -- ICU transfer (acted rows only) | 0.9431 |
| F1 -- macro (acted rows only) | 0.9711 |

F1 is computed only on the 12,691 rows where the agent predicted discharge or transfer (not wait). The 59.4% wait rate at terminal rows means the agent deferred on more than half of stays even at the true decision point. Among the stays where it did commit, accuracy is very high.

### First-Commit Metrics

For each test stay, the first event where the agent predicted discharge (0) or transfer (1) is identified as the agent's commitment point.

| Metric | Value |
|---|---|
| Test stays | 31,267 |
| Stays with a commit | 14,470 (46.3%) |
| Stays with no commit | 16,797 (53.7%) |
| Commit action = discharge | 14,262 (98.6% of committed) |
| Commit action = transfer | 208 (1.4% of committed) |
| True label = discharge (among committed) | 14,130 (97.6%) |
| True label = ICU (among committed) | 340 (2.4%) |
| F1 -- discharge at first commit | 0.9944 |
| F1 -- ICU at first commit | 0.7117 |
| F1 -- macro at first commit | 0.8531 |
| Accuracy at first commit | 0.9891 |
| Median commit event index | 6 |
| Median provider event index | 7 |
| Agent committed earlier than provider | 3,537 stays (24.4%) |
| Agent committed same event as provider | 10,933 stays (75.6%) |
| Agent committed later than provider | 0 stays (0.0%) |
| Median steps earlier (among earlier commits) | 4 events |

**Notes:**
- 53.7% of test stays receive no commit, meaning the agent chose `wait` on every event of those stays. This reflects that `wait` was the action for all non-terminal rows in training, and the agent learned to be conservative about committing.
- Among committed stays, the agent is heavily biased toward discharge (98.6%) vs. ICU transfer (1.4%), while the true ICU rate among committed stays is 2.4%. The agent's lower ICU commit rate contributes to the lower ICU F1 (0.7117 vs. discharge F1 of 0.9944).
- With corrected `p_icu` ordering, the median commit event shifted to 6 (vs. provider median of 7), and early commits increased to 24.4% of committed stays (up from 13.7% in the previous run on misaligned data). The agent no longer commits later than the provider on any stay.
- Terminal row F1 (macro 0.9711) is higher than first-commit F1 (macro 0.8531), confirming that early commitment costs some accuracy on ICU identification, primarily in recall (0.5735 vs. higher recall at the terminal row).
