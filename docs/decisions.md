# Who our patient cohort is

The cohort consists of adult ED patients from MIMIC-IV (Beth Israel Deaconess Medical Center) whose ED visit ended in one of two terminal outcomes: discharge directly from the emergency department, or transfer to an ICU. Patients admitted to a general hospital ward without ICU transfer are excluded -- the clinical focus of this project is early identification of ICU-level acuity from data available in the first hour of an ED visit. Patients who died in the ED are also excluded.

Stays with an invalid or inverted stay window (see Stay Window Correction) and stays with fewer than two time steps after window filtering are also dropped.

## Target Variable

**Original design:** The project was initially designed as a multi-label action prediction problem, predicting the next clinical intervention among 7 discrete action types: lab order, culture order, administer medication, start medication, stop medication, rate change, and observe (no action). The intent was to train an offline RL agent to learn a treatment policy by imitating historical clinician behavior.

**Revised design:** The multi-label approach was abandoned for two reasons:
1. **Data sparsity:** Most action types occur infrequently within any given time window, producing a heavily zero-inflated target matrix where the majority of rows have no positive label for most action types.
2. **Class imbalance:** The observe/no-action label dominated all others by a large margin, making it difficult for models to learn meaningful distinctions between rarer action types.

The supervised learning target was revised to a binary outcome: **terminal status** -- discharge (0) vs. ICU transfer (1). This is derived from the `terminal_event` column in the cohort and encoded as `label` (0 = discharge, 1 = transfer_icu) in the modeling pipeline. Predicting terminal outcome from early ED data is a well-defined, clinically meaningful problem with a stable class distribution.

The RL agent retains an action framework, but the action space was redesigned from the original multi-label formulation -- see Offline RL Agent and Scrapped Designs sections.

# How we define and set up the state object

## Microbiology / Culture State Features

**Original design:** Cultures were initially represented as a single binary feature -- `culture_ordered` (binary: was any culture ordered during this ER visit?). A time-decay feature on the order time was deferred to the feature engineering phase.

**Current design:** Microbiology state is encoded as OHE with four binary status columns per specimen type, mirroring the lab OHE approach. The top 20 specimen types each receive four columns: `{spec_type}_Pending`, `{spec_type}_Positive`, `{spec_type}_Negative`, `{spec_type}_Other`. All remaining specimen types are bucketed under `OTHER`.

**Rationale for change from binary to OHE:**
1. **Model performance:** The binary `culture_ordered` feature loses clinically relevant information about result status. A pending culture (unknown result) and a confirmed negative culture are meaningfully different patient states that the binary encoding collapses into the same value.
2. **Label distance:** Encoding status as an ordinal (0 = not ordered, 1 = pending, 2 = positive, 3 = negative) implies equal semantic distances between adjacent values that do not exist. OHE eliminates this false distance assumption, allowing the model to treat each status independently.

`culture_positive` is retained for post-analysis but is NOT a real-time state feature -- only ~2% of culture results were available before ED discharge.

### `culture_result` labeling

A `culture_result` column is derived from `org_name` and `comments` using a regex-based classification function. `org_name` is checked first; if null, `comments` is checked. The four result labels are:

- **POSITIVE** -- `org_name` contains an organism name (no negative keyword present); or `comments` contains colony counts, growth language, contamination, or other clinical indicators of a positive result
- **NEGATIVE** -- `org_name` or `comments` contains language indicating no organism detected: no growth, not detected, negative, nonreactive, indeterminate, or no [organism] seen/found/isolated
- **CANCELLED** -- `org_name` or `comments` indicates the test was not completed: cancelled, test not performed, or patient credited
- **OTHER** -- both `org_name` and `comments` are null, or `comments` contains only placeholder characters (underscores, dashes) with no interpretable content

`org_name` alone is not a fully reliable indicator -- some positives are documented only in free-text `comments`. The classifier handles both columns to maximize coverage without manual review.

## What the time step is

Time steps are event-driven, not fixed-interval. A time step is created at each unique event time within a patient's stay window. Events that generate time steps include: vital sign charting, lab result availability, medication dispensing, microbiology culture orders, ECG results, and radiology results. The union of all event times within the stay window is sorted chronologically and assigned a sequential `step_idx` starting at 0.

The `time` column records the absolute timestamp of each step. `time_since_last_min` records elapsed minutes since the previous step (0 at the first step of each stay). `stay_window_start` and `stay_window_end` define the bounds of the stay.

State feature values carry forward from their last recorded value until updated -- if no new lab results arrive between steps 3 and 7, the lab state columns at step 7 still reflect the results from step 3. This represents the clinician's best current knowledge at each moment.

Stays with only a single time step after window filtering are dropped -- a minimum of two steps is required for temporal features to be meaningful.

## Training Data Structure

The full patient state dataset (`full_patient_state`) contains one row per `(ed_stay_id, time)` pair, representing the complete patient state at each event-driven time step. Each row captures static patient attributes, current vital signs, and cumulative knowledge of all clinical events that have occurred up to that point in the stay.

**Column groups:**

| Group | Description |
|---|---|
| Identifiers | `ed_stay_id`, `subject_id`, `hadm_id`, `time`, `step_idx`, `stay_window_start`, `stay_window_end` |
| Static patient | `gender`, `anchor_age`, `race`, `acuity`, `chiefcomplaint`, `height`, `weight`, arrival transport OHE columns |
| Current vitals | `current_temperature`, `current_heartrate`, `current_resprate`, `current_o2sat`, `current_sbp`, `current_dbp`, `current_pain`, `current_map` |
| Temporal vitals | `_rolling1h`, `_delta`, `_rate_per_min` suffix variants for each vital column |
| Lab state (OHE) | `{category}-{fluid}_Normal`, `{category}-{fluid}_Pending`, `{category}-{fluid}_Abnormal` per lab type (19 category × fluid combos = 57 columns) |
| Micro state (OHE) | `{spec_type}_Pending`, `{spec_type}_Positive`, `{spec_type}_Negative`, `{spec_type}_Other` for top 20 specimen types + OTHER (84 columns) |
| Dispensed medications | Binary flag per drug class (ACE Inhibitor through Other) |
| Pre-arrival medications | `recon_*` binary flags from medication reconciliation at ED intake |
| ECG / Radiology | OHE status columns for ECG and radiology result categories |
| Action flags | `vitals_checked`, `labs_ordered`, `micro_ordered`, `ecg_ordered`, `rad_ordered`, `dispense_meds`, `ward_transfer`, `in_ed`, `in_ward` |
| Target | `terminal_event` (string), `terminal_code` (0/1), `cohort_label`, `total_length` |

State columns carry forward from their last updated value -- each row reflects cumulative knowledge at that point in the stay, not only events that occurred at that exact timestamp.

## Dispensed Medications State

Medications dispensed during the ED visit and hospital stay are represented as binary flags per drug class. The drug class taxonomy used is the dispensed medication categories from the `ed.pyxis` and `hosp.emar` tables, collapsed to the top-level drug class for each dispense event. Each column takes value `1` if any medication in that class has been dispensed up to the current time step, `0` otherwise.

The column range runs from `ACE Inhibitor` through `Other`, covering all drug classes present in the cohort. The exact set of columns is derived from the data at pipeline time rather than hardcoded, to accommodate any changes in the underlying dispense records.

## ECG and Radiology State

ECG and radiology results are encoded as OHE status columns per result category. Each category gets a binary column for each possible status (e.g., normal, abnormal, pending). As with labs and microbiology, the OHE encoding was chosen over ordinal to avoid imposing false distances between result statuses.

ECG columns are prefixed `ecg_status_` and radiology columns are prefixed `rad_status_`.

## Chief Complaint Encoding

Chief complaint (`chiefcomplaint`) is a free-text field recorded at triage. Two encoding strategies are used depending on the model:

- **Logistic Regression and Random Forest:** The raw text is mapped to one of 19 clinical categories (e.g., `chest_pain`, `dyspnea`, `altered_mental_status`) using regex pattern matching, then OHE-encoded. The categories are ordered by specificity -- the first matching pattern wins. Complaints that match none of the 19 categories are assigned `other`.

- **XGBoost:** TF-IDF encoding (top 50 terms, unigrams and bigrams, stop words removed) is applied after the train/test/validation split. The TF-IDF vectorizer is fit on the training set only and then applied to test and validation to prevent leakage.

The regex category approach captures the clinical concepts most relevant to ED acuity decisions. TF-IDF is used for XGBoost to leverage its ability to handle high-dimensional sparse inputs more effectively than tree splits on OHE categories.

## Pre-Arrival Medications (Medrecon)

Pre-arrival medication reconciliation data from `ed.medrecon` is included as static binary flags per drug class (`recon_*` columns). This captures the patient's home medication list as documented at ED intake -- a strong signal of chronic conditions and baseline health status.

Since medrecon is recorded once per visit (not time-varying), the same flags are broadcast to all time steps of each stay via a left merge on `ed_stay_id`.

# Outlier Handling for Stay Length

Stays in the extreme tail of the length-of-stay distribution are removed from the modeling pipeline. Exceptionally long hospital stays introduce non-representative trajectories driven by chronic care rather than acute ED decision-making.

Two hard-coded thresholds are applied in `modeling/data_prep/lstm.py`:

- **`pad_length = 100`**: Stays with more than 100 events (clinical actions/timesteps) are dropped. Because `full_patient_state` is event-driven -- each row corresponds to a discrete clinical event, not a fixed time interval -- this caps the number of events per stay, not elapsed time. Zero-padding all sequences to a uniform length of 100 is the memory management strategy for the LSTM. A percentile-based cutoff would have retained stays up to 300-400 events, making padding impractical to plan for and significantly increasing batch memory requirements.
- **`stay_length = 100`**: Stays with total hospital LOS > 100 days are dropped. Approximately 9,500 records exceeded this threshold.

Hard cutoffs were preferred over percentile-based ones because padding requires knowing the fixed maximum length in advance. A percentile cutoff varies by dataset version and complicates memory planning.

# Cohort DF EDA/Prep

## Triage Outliers/Missing

Triage vitals in the MIMIC-IV ED dataset contain a significant number of implausible values, likely due to data entry errors (extra digits, wrong units, transposed values). The strategy is to correct values where a clear pattern exists, null out values where no reliable correction is possible, and impute remaining nulls in the subsequent pipeline step (ffill + mean imputation). Corrections are applied in a fixed order within each variable because some transforms create values that are then caught by a downstream rule.

### Temperature

Normal range: 97--99°F. Hypothermia below 95°F, severe below 82°F. Hyperthermia above 100°F, severe above 104°F. Recorded values extend up to 110--115°F in rare cases.

| Rule | Action | Assumption |
|---|---|---|
| Value > 900 | Divide by 10 | Extra digit entered (e.g. 986 → 98.6) |
| 28 < value ≤ 40 | Convert Celsius → Fahrenheit: `(x * 1.8) + 32` | Value was recorded in Celsius instead of Fahrenheit |
| 5 < value < 10 | Multiply by 10 | Missing leading digit (e.g. 9.8 → 98) |
| Value > 115 | Set to null | No recoverable correction pattern; imputed in subsequent step |

Values below 82°F that survive all transforms (e.g. the 46--80°F range) are left as-is and will be addressed in a downstream validation step or absorbed by the imputation pipeline. There is not enough confidence to correct or null these without additional context.

### Heart Rate

Normal range: 60--100 bpm. Tachycardia above 100, bradycardia below 60. Values below 20 are inconsistent with a living patient at triage.

| Rule | Action | Assumption |
|---|---|---|
| Value > 500 | Divide by 10 | Extra digits entered (e.g. 800 → 80) |
| Value < 20 | Set to null | No clear correction pattern; imputed in subsequent step |

Values in the 256--500 range are left as-is. Spot-checking suggests most are plausible extreme tachycardia, and there is no consistent correction pattern for the minority that may be errors.

### Respiratory Rate

Normal range: 12--20 breaths/min. Above 20--25 can indicate tachypnea.

| Rule | Action | Assumption |
|---|---|---|
| Value > 1000 | Divide by 100, round | Two extra digits entered (e.g. 1800 → 18) |
| 100 < value ≤ 1000 | Divide by 10, round | One extra digit entered (e.g. 180 → 18) |
| Value < 4 | Set to null | No clear correction pattern; imputed in subsequent step |

The divide-by-100 rule is applied before divide-by-10 to avoid double-correcting values above 1000.

### O2 Saturation

Normal range: 95--100%. Clinically significant drop below 88%.

| Rule | Action | Assumption |
|---|---|---|
| 900 < value < 1010 | Divide by 10, floor | Extra digit entered (e.g. 980 → 98, 1000 → 100) |
| 0 < value ≤ 10 | Add 90 | Missing leading digit (e.g. 8 → 98) |
| value == 0, value > 100, or value < 40 | Set to null | Not recoverable; imputed in subsequent step |

Values in the 40--88 range are left as-is -- many in this range are clinically plausible (severely hypoxic patients), and there is not enough confidence to null them without individual review.

### Systolic Blood Pressure

Normal range: 90--120 mmHg. Elevated 120--140, hypertensive 140+. Critically low below 50--60.

| Rule | Action | Assumption |
|---|---|---|
| Value > 270 | Set to null | Spot-checking found no consistent correction pattern above this threshold |
| Value < 40 | Set to null | Too low to be plausible at triage |

Values between 40 and 270 are retained. There is a cluster near 24--40 that is suspicious but enough values in that range appear accurate upon spot-checking to avoid a blanket null.

### Diastolic Blood Pressure

Normal range: 60--80 mmHg. Elevated 80--120+. Values below 50 start to get low.

| Rule | Action | Assumption |
|---|---|---|
| Value > 150 | Set to null | Likely charting errors with no clear correction pattern |
| Value < 20 | Set to null | Too low to be plausible |

## Stay Window Correction (Inverted Times)

A small number of records had `stay_window_end` before `stay_window_start`, producing negative `time_steps`. The common pattern among these records was in-hospital death -- discharge times were recorded in a disjointed way relative to the ED arrival, causing the window to appear inverted.

**Fix:** For each affected record, `stay_window_end` was reset to `ed_outtime` (the ED departure timestamp). These corrected times were saved to `df_grouped.csv` and applied before `time_steps` is calculated.

A small number of records (~5) had no valid stay window after correction and were dropped from the cohort.

## Race

The raw `race` column contains 33 distinct categories, which is too high a cardinality to use directly as a state feature. The column is collapsed to 6 categories using regex matching on the original string values.

**Collapsed categories:**

| Category | Covers |
|---|---|
| `White` | WHITE, WHITE - OTHER EUROPEAN, WHITE - EASTERN EUROPEAN, WHITE - RUSSIAN, WHITE - BRAZILIAN, PORTUGUESE |
| `Black` | BLACK/AFRICAN AMERICAN, BLACK/AFRICAN, BLACK/CAPE VERDEAN, BLACK/CARIBBEAN ISLAND |
| `Hispanic` | HISPANIC OR LATINO, all HISPANIC/LATINO - * subcategories, SOUTH AMERICAN |
| `Asian` | ASIAN, all ASIAN - * subcategories, NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER |
| `Native American` | AMERICAN INDIAN/ALASKA NATIVE |
| `Other` | OTHER, MULTIPLE RACE/ETHNICITY, UNKNOWN, PATIENT DECLINED TO ANSWER, UNABLE TO OBTAIN, NaN |

**Rationale for Unknown/Declined → Other:** These patients have a race -- we simply don't know what it is. Creating a separate `Unknown` category would imply a meaningful distinction that doesn't exist. Collapsing them into `Other` is more honest about what the data actually represents.

## Pain

The `pain` column is a 0--10 self-reported pain scale recorded at triage. Missing or non-numeric values are coerced to NaN, a `pain_missing` indicator is created, and the column is then imputed alongside the other numeric vitals (see Missing Value Imputation Strategy).

**Main issues:**
- Mixed types: the column contains integers, floats, free-text strings, and punctuation artifacts (quotes, `>`, `-`, `+`)
- Range entries: some values are entered as a range (e.g. `5-7`) rather than a single number
- Values above 10: present in significant volume; could be mis-entries, carryovers from another field, or a different scale (e.g. Glasgow Coma Scale, which would partially explain the large cluster at 13)
- Float values: the scale is 0--10 integers; floats need to be rounded

| Step | Action | Assumption |
|---|---|---|
| Strip punctuation and normalize case | Remove `"`, `"`, `"`, `>`, `-`, `+` from start/end; lowercase | These are data entry artifacts, not meaningful values |
| Range entries `#-#` | Replace with first (lower) number | The lower end of a reported range is the more conservative estimate |
| Coerce to numeric | Non-numeric strings → NaN | Free-text entries are not recoverable as a pain score |
| Values > 10 | Set to NaN | Not a valid 0--10 scale entry; source is ambiguous and not correctable |
| Float values | Round to nearest integer | Pain scale is whole numbers only |
| Remaining NaN | Create `pain_missing = 1`; impute with column mean | Originally masked as `'Other'` as a placeholder; revised to follow the same ffill + mean imputation pipeline as other numeric vitals. A `pain_missing` indicator preserves the information that pain was unobserved. |

# Vitals EDA/Prep

## Rhythm Column

The `rhythm` column (cardiac rhythm, e.g. "Sinus Rhythm", "Atrial Fibrillation") is dropped from the vitals dataframe. The column is ~96% missing across the dataset, making it impossible to impute meaningfully or use as a reliable state feature.

## Triage Row Injection

Before imputation, the triage vitals from the cohort dataframe (`cohort_with_triage`) are injected as a synthetic first reading for each ED stay. The triage row uses `ed_intime` as its `charttime` and carries the triage values for all numeric vital columns. This serves two purposes:
1. Anchors the time series at the start of the stay so ffill has a starting value to propagate
2. Ensures patients who have triage vitals but no charted vitals readings still contribute imputable data

## Missing Value Imputation Strategy

Imputation is applied in two passes in order:

1. **Forward fill within stay** -- sorts by `(ed_stay_id, charttime)` and propagates the last known reading forward within each stay. Values never bleed across stays. The last recorded value is the best real-time estimate of current patient state.
2. **Column mean imputation** -- any NaN remaining after ffill (stays with no reading at all for a column, or leading NaNs before the first observed value) are filled with the column mean.

**Why mean imputation over KNN or IterativeImputer:** Correlation analysis across vital sign columns found no meaningful correlations between vitals at the population level. KNN and iterative imputation exploit inter-feature correlations to produce better-than-mean fills -- without those correlations, they converge to the column mean anyway while adding complexity. Mean imputation was chosen to avoid artificially shifting the distribution and to keep the pipeline simple and interpretable.

Rhythm is excluded from this pipeline -- it was dropped entirely; see Rhythm Column above. Pain follows the same imputation pipeline as the numeric vitals; see Pain above.

## Feature Engineering

After imputation, the following features are derived from the numeric vital columns (`temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`):

### Time-based features (per stay, sorted by charttime)

| Feature | Formula | Notes |
|---|---|---|
| `time_since_last_hrs` | `diff(charttime)` in hours | Hours since previous reading within the stay. First reading of each stay is NaN. Also a feature in its own right -- frequent readings signal closer monitoring. |
| `{col}_rolling1h` | 1-hour trailing rolling mean | Time-based window (not row-based) so irregular sampling is handled correctly. Computed per stay using `set_index('charttime').rolling('1h')`. |
| `{col}_delta` | `diff()` of column values | 1-step change from previous reading. First reading of each stay is NaN. |
| `{col}_rate_per_min` | `{col}_delta / time_since_last_min` | Rate of change per minute. Normalises delta for the time gap between readings. |

### Blood pressure derived features

| Feature | Formula | Clinical meaning |
|---|---|---|
| `map` | `dbp + (sbp - dbp) / 3` | Mean Arterial Pressure -- standard clinical measure of average perfusion pressure. Often more predictive of organ perfusion than sbp or dbp alone. |

# Lab Event EDA/Prep

## Lab Event Simplification

The raw lab events table contains one row per individual lab test result, meaning a single lab order batch (e.g. a CBC) can produce 20+ rows for the same patient at the same timestamp. To simplify the action space and reduce redundancy, the table is collapsed to one row per `(ed_stay_id, category, fluid, order_time)` group.

**Action space:** Instead of using individual lab `label` as the action (high cardinality), we use the unique combination of `category` and `fluid`. This yields 19 possible lab actions the agent can take, which is tractable and clinically meaningful -- e.g. ordering Hematology/Blood is a distinct decision from ordering Chemistry/Urine.

**Collapsing logic:**
- One row per `(ed_stay_id, category, fluid, order_time)` -- this represents a single lab action at a point in time
- `abnormal`: `True` if any individual result in the group had a non-null flag, `False` otherwise. Non-null flag = abnormal result; null = normal.
- `result_time`: max across the group -- represents when all results in the batch are back
- `subject_id`, `hadm_id`, `ordered_location`: same within a group, take first value

**Example:** A patient with 24 Hematology/Blood labs ordered at the same timestamp collapses to a single row. If any of those 24 came back abnormal, `abnormal=True`.

# full_patient_state Dataset

The `full_patient_state` dataset is the output of the `data_pipelines/combine_patient_state/` pipeline and is the primary input to all modeling work. It is stored on HuggingFace under `ADS599-Capstone/modeling_data` (config: `full_patient_state`, split: `full_patient_state`).

**Build pipeline:** `python -m data_pipelines.combine_patient_state.main`

The pipeline loads all intermediate feature datasets from HuggingFace (`interim_data` repo), joins them on `ed_stay_id` at each event time step, and produces a single wide DataFrame. The major steps are:

1. Load cohort and build `terminal_event` label
2. Collect all event timestamps and build the step index
3. Drop out-of-window rows and single-step stays
4. Snap vitals to each time step, compute `time_since_last_min` and rolling/delta/rate features
5. Expand lab OHE columns (`_Normal`, `_Pending`, `_Abnormal` per category × fluid)
6. Expand microbiology OHE columns (`_Pending`, `_Positive`, `_Negative`, `_Other` per specimen type)
7. Expand dispensed medication flags by drug class
8. Snap ECG and radiology OHE status columns
9. Add location flags, height/weight, and medrecon flags
10. Add derived features (arrival OHE, acuity ffill, total length, drop non-feature columns)
11. Run validation checks and push to HuggingFace

A checkpoint file (`_checkpoint.parquet`) is saved after each major step so a failed run can resume from the last completed step rather than restarting from scratch.

# Modeling Approach

Three complementary modeling approaches are used, each operating on different representations of the same underlying data.

## Traditional ML (Logistic Regression, Random Forest, XGBoost)

These models operate on a **one-row-per-stay** aggregation of `full_patient_state`, using only data from the first 60 minutes of each ED visit. The aggregation uses triage-time vital values (`first` within the window) and worst-case flags for binary event columns (`max` within the window). This produces a single feature vector per stay suitable for standard supervised learning.

**Purpose:** These models were trained alongside the LSTM to evaluate whether a strong predictive signal for patient disposition (discharge vs. ICU transfer) exists in the available ED data. The focus is signal detection -- can the outcome be predicted from early ED data? -- which is why this setup differs from the RL agent formulation.

**Data preparation:** `modeling/data_prep/traditional_ml.py`

**Train/test/validation split:** 80/10/10, split by `subject_id` (not by `ed_stay_id`) to prevent data leakage across multiple visits by the same patient. Stratified on the binary label to preserve class balance across splits.

**Scaling:** `StandardScaler` fit on the training set, applied to test and validation. Scaling columns are defined in `modeling/config/traditional_ml.yaml`.

**Chief complaint:** OHE category encoding for Logistic Regression and Random Forest; TF-IDF (top 50 terms, fit on train only) for XGBoost.

**Class imbalance:** Logistic Regression and Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight` calculated from the training set class ratio (negative count / positive count).

**Scripts:** `modeling/traditional_ml/{logistic_regression,random_forest,xgboost_train}.py`

**Artifacts:** Saved to `modeling/artifacts/{log_reg,random_forest,xgboost}/` -- model pickle, scaler pickle, test and validation parquets.

## LSTM

The LSTM operates on the full event-driven time series from `full_patient_state`, using variable-length sequences per stay. This allows the model to capture temporal patterns in vital sign trends, lab result timing, and treatment sequences that the one-row aggregation discards.

**Purpose:** Like the traditional ML models, the LSTM was trained to determine whether a strong signal for patient disposition exists in the sequential ED data. The LSTM's additional value is that it can exploit trajectory information -- not just what the patient's state is at a point in time, but how it has evolved over the stay.

**Architecture:** 2-layer LSTM, `hidden_size = 256`, `dropout = 0.2`. The final hidden state feeds a linear output layer with 2 classes (discharge / ICU transfer).

**Training:**
- Loss: CrossEntropy with class weights derived from the training set class ratio to counteract class imbalance
- Optimizer: AdamW, `lr = 0.0001`, `weight_decay = 0.1`
- Early stopping: patience = 3 on validation loss, tolerance = 0.001
- Sequence handling: zero-padded to `pad_length = 100` events; packed sequences used during forward pass so padding is masked out and does not contribute to loss

**Train/val/test split:** Same 80/10/10 by `subject_id` as traditional ML.

**Scaling:** `StandardScaler` fit on the training set only, applied to val/test.

**Scripts:** `modeling/lstm/train.py`

**Artifacts:** Saved to `modeling/artifacts/lstm/` -- model weights, scaler pickle, test parquet, metrics CSV.

## Step-by-Step LSTM Inference and `sbs_predictions`

After training the standard LSTM, a separate inference pass (`modeling/lstm/step_by_step_train.py`) runs the model at every individual event timestep of every stay and records the per-step ICU transfer probability as `p_icu`. This produces the `sbs_predictions` dataset, pushed to HuggingFace under `ADS599-Capstone/modeling_data`.

**Why a separate split:** The RL agent cannot train on `p_icu` values generated from the same data the LSTM trained on -- predictions on training data are overfit and would give the RL agent a misleading view of model confidence. The step-by-step inference uses a different split: `train_size = 0.6` (vs. 0.8 for the standard LSTM), giving the RL agent a larger pool of held-out predictions to learn from. The RL agent needs more samples than a supervised model to learn a meaningful policy from offline data.

**What `sbs_predictions` contains:** All feature columns from `full_patient_state`, plus:
- `p_icu` -- LSTM step-by-step ICU transfer probability at each event
- `delta_p_icu` -- change in `p_icu` from the prior event, capturing momentum in model confidence (a rising trend is a different clinical signal from a plateau, even at the same current value)
- `event_idx` -- within-stay event counter
- `action_taken` -- 0 (discharge) or 1 (transfer_icu) at terminal rows, 2 (wait) at all other rows
- `reward` -- computed from the reward function at each step

The RL agent's state input is built from `sbs_predictions`, not directly from `full_patient_state`.

**Artifacts:** Saved to `modeling/artifacts/step_by_step/`.

## Offline RL Agent

The RL agent learns a disposition escalation policy from historical clinician behavior using offline DQN with Conservative Q-Learning (CQL). The agent observes the patient state at each event and selects from a **3-action space**:

| Action | Code | Meaning |
|---|---|---|
| discharge | 0 | Patient can be safely discharged from the ED |
| transfer_icu | 1 | Patient requires ICU-level care |
| wait | 2 | Insufficient information to commit to a disposition yet |

In the historical data, non-terminal rows have `action_taken = 2` -- the provider continued monitoring without making a final disposition. Terminal rows have `action_taken = label` -- the provider's actual disposition. This reflects reality: the provider made an active discharge/transfer decision exactly once per stay.

The `wait` action is essential for two reasons. First, without it the agent is forced to predict discharge or transfer at every timestep, which is clinically meaningless for early events where no decision has been made. Second, `wait` functions as a placeholder for all other clinical actions (lab orders, medication changes, etc.) currently outside the project scope, preserving the ability to expand the action space in future work.

**Architecture:** `ProviderNetwork` -- a feedforward DQN with layers 256 → 256 → 128 → 3 (ReLU activations, dropout = 0.1). Three output units are the Q-values for discharge, transfer_icu, and wait.

**Conservative Q-Learning (CQL):** Offline RL cannot explore the environment, so Q-values for out-of-distribution (OOD) actions can be overestimated without penalty. CQL adds a regularization term to the training loss:

```
loss = MSE(Q_taken, bellman_target) + cql_weight * (logsumexp(Q_all, dim=1) - Q_taken).mean()
```

The logsumexp formulation is always ≥ 0 by Jensen's inequality, and approaches 0 when `Q_taken = max(Q_all)`. This penalizes the model for assigning high Q-values to actions unsupported by the data without over-penalizing well-supported actions.

The logsumexp form was adopted after adding the `wait` action. With ~99% of rows having `action_taken = wait`, the earlier `Q_all.mean() - Q_taken.mean()` formulation became massively negative: `wait`'s Q-value was trained on real data while discharge/transfer Q-values were OOD extrapolations, so the mean of all Q-values fell far below `Q_taken`. This flipped the loss sign, caused Q-value collapse, and sent training in the wrong direction. Logsumexp avoids this by guaranteeing non-negativity.

**Target network:** A separate frozen copy of the network (`target_network`) provides stable Bellman targets. Without this, the training network would chase its own shifting Q-value estimates on every gradient step, making learning unstable. The target network's weights are hard-synced from the training network every `target_update_freq = 500` steps, giving it time to provide stable targets before updating.

**Bellman target:**
```
Q_target(s, a) = r + gamma * max_a' Q_target(s', a') * (1 - is_terminal)
```
`gamma = 0.9`. At terminal rows, the bootstrapped future term is zeroed out.

**Training config:**
- Optimizer: AdamW, `lr = 0.00005`, `weight_decay = 0.01`
- Gradient clipping: `max_norm = 1.0`
- `cql_weight = 0.1`

**Train/test split:** 80/20 by `ed_stay_id` within `sbs_predictions`.

**Scripts:** `modeling/rl_agent/train.py`

**Artifacts:** Saved to `modeling/artifacts/rl_policy/` -- policy weights (`.pt`), fitted scaler (`.pkl`), epoch metrics (`.csv`).

## RL State Features

The RL agent observes a subset of columns from `sbs_predictions`. State columns are defined in `modeling/config/rl_agent.yaml` and normalized with a `StandardScaler` fit on the RL training split. Key features:

- `p_icu` -- LSTM step-by-step ICU transfer probability at the current event
- `delta_p_icu` -- change in `p_icu` from the prior event (rising vs. falling confidence are different signals even at the same current value)
- `event_idx` -- within-stay event counter (tells the agent how far into the stay it is)
- Vital signs: `anchor_age`, `acuity`, `current_temperature`, `current_heartrate`, `current_resprate`, `current_o2sat`, `current_sbp`, `current_dbp`, `current_pain`, `current_map`, `height`, `weight`

# Rewards

## Reward Function

The reward function provides both a terminal outcome signal and a per-step shaping signal.

**Terminal reward:**

At the stay's terminal event the reward is confidence-weighted by the LSTM's ICU probability at that moment (`p_icu`):

| Action | True outcome | Reward |
|---|---|---|
| discharge | discharge | `+base_reward * p_icu` |
| transfer_icu | transfer_icu | `+base_reward * class_ratio * p_icu` (minority class bonus) |
| discharge | transfer_icu | `-base_reward * class_ratio * p_icu` (missed ICU -- heaviest penalty) |
| transfer_icu | discharge | `-base_reward * p_icu` |

`base_reward = 0.4`, `class_ratio = 2.0`. The class ratio bonus and penalty apply to ICU-related outcomes to counteract class imbalance -- correct ICU identification is rewarded more, and a missed ICU transfer is penalized more heavily than an incorrect transfer decision.

**Intermediate reward:**

Every event receives a small negative event-count-based penalty:

```
reward += event_idx * time_weight    (time_weight = -0.001)
```

This penalizes longer stays by accumulating a larger penalty per event index, nudging the agent toward earlier decisions. The penalty is event-count-based rather than time-based because the state is event-driven -- irregular sampling intervals make time-based penalties less interpretable and harder to compare across stays.

# Evaluation

## Traditional ML and LSTM

Traditional ML models and the LSTM are evaluated on the held-out test split using:

- **PR-curve (precision-recall)** -- preferred over ROC-AUC as the primary metric for this dataset because of class imbalance. ROC-AUC can appear high even when minority-class (ICU transfer) recall is poor; PR-curve surfaces this directly.
- **ROC-AUC** -- reported alongside PR-curve for completeness
- **F1 by class** -- separately for discharge and ICU transfer
- **Confusion matrix** -- normalized, to show per-class error rates independently of class size
- **Probability distribution** -- histogram of predicted probabilities per true class, to assess class separation and model confidence
- **Calibration curve** -- predicted probability vs. observed fraction positive, to assess how close the model's confidence is to the true likelihood of ICU transfer
- **Performance by stay length** -- F1 by stay-length quartile, to understand whether shorter stays (less available information) degrade prediction quality

## RL Agent

The RL agent is evaluated on the test split of `sbs_predictions` using:

- **Terminal-row action distribution** -- at the stay's true decision point, what fraction of agent predictions were wait / discharge / transfer? Broken down by true disposition (discharge vs. ICU transfer).
- **F1 at terminal rows** -- on terminal rows where the agent acted (predicted discharge or transfer rather than wait), F1 for each class
- **First-commit timing** -- for each test stay, the first event where the agent predicts discharge or transfer. Compared to the provider's decision event index to measure how many events earlier the agent commits.
- **F1 at first commit** -- comparing F1 at the agent's first commit vs. F1 at the terminal row quantifies whether committing earlier costs accuracy.
- **By stay length** -- F1 and median steps-earlier broken down by stay-length quartile, to assess whether early detection is more or less feasible for shorter stays.

# Scrapped Designs

## Original Multi-Label Action Space

The original RL formulation treated the problem as multi-label action prediction: at each timestep the agent would predict the next clinical intervention from a space of 8--10 concurrent actions derived directly from the data. This included:

**Medication actions (4 types):** Derived from `event_txt` in `hosp.emar` and `ed.pyxis`:

| Action | emar `event_txt` values mapped |
|---|---|
| `administer_medication` | Administered, Partial Administered, Delayed Administered, Administered Bolus from IV Drip, Administered in Other Location |
| `start_medication` | Started, Restarted |
| `stop_medication` | Stopped, Stopped As Directed, Stopped - Unscheduled, Stopped in Other Location |
| `rate_change` | Rate Change |

**Lab actions (19 types):** One per unique combination of `category` × `fluid` present in the ED cohort. Collapsed from individual test-level labels to reduce the action space from 35 to 19 while preserving clinically meaningful distinctions (e.g. Hematology/Blood vs. Chemistry/Urine are genuinely different ordering decisions).

**Microbiology actions:** One per top-20 specimen type by frequency, with remaining types bucketed as `OTHER`.

This design was abandoned for two reasons:
1. **Data sparsity:** Most action types were zero for most timesteps, creating a heavily zero-inflated target matrix where one action (observe/no-action) dominated all others by a large margin.
2. **Signal difficulty:** With one action dominating, the agent could not learn when to take the rarer actions or distinguish meaningful patterns among them.

## Intermediate Iteration: Collapsed Action Categories

After abandoning individual action prediction, an intermediate design was evaluated: retaining the same clinical events as state features in the data but collapsing the action space to high-level categories -- `order_labs`, `order_cultures`, `dispense_meds`, etc. This reduced sparsity somewhat but the fundamental problem remained: the dominant "no action" category still overwhelmed everything else, and getting a meaningful signal from this many simultaneous decision types would have required a much larger dataset and more training infrastructure than the project scope allowed.

## Scaling to Escalation Decisions

The project scope was refined to focus on a single clinically impactful decision: **when to escalate care or discharge the patient**. This maps cleanly to the 3-action RL formulation (discharge / transfer_icu / wait) and aligns the RL objective directly with the supervised learning target. The `wait` action captures all other clinical activity (ordering labs, adjusting medications) as a single "not yet decided" signal, with the intent that the action space can be expanded in future work as more training data and compute become available.