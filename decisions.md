# Who our patient cohort is

The cohort consists of adult ED patients from MIMIC-IV (Beth Israel Deaconess Medical Center) whose ED visit ended in one of two terminal outcomes: discharge directly from the emergency department, or transfer to an ICU. Patients admitted to a general hospital ward without ICU transfer are excluded -- the clinical focus of this project is early identification of ICU-level acuity from data available in the first hour of an ED visit. Patients who died in the ED are also excluded.

Stays with an invalid or inverted stay window (see Stay Window Correction) and stays with fewer than two time steps after window filtering are also dropped.

## Target Variable

**Original design:** The project was initially designed as a multi-label action prediction problem, predicting the next clinical intervention among 7 discrete action types: lab order, culture order, administer medication, start medication, stop medication, rate change, and observe (no action). The intent was to train an offline RL agent to learn a treatment policy by imitating historical clinician behavior.

**Revised design:** The multi-label approach was abandoned for two reasons:
1. **Data sparsity:** Most action types occur infrequently within any given time window, producing a heavily zero-inflated target matrix where the majority of rows have no positive label for most action types.
2. **Class imbalance:** The observe/no-action label dominated all others by a large margin, making it difficult for models to learn meaningful distinctions between rarer action types.

The supervised learning target was revised to a binary outcome: **terminal status** -- discharge (0) vs. ICU transfer (1). This is derived from the `terminal_event` column in the cohort and encoded as `label` (0 = discharge, 1 = transfer_icu) in the modeling pipeline. Predicting terminal outcome from early ED data is a well-defined, clinically meaningful problem with a stable class distribution.

The RL agent retains the action framework but uses terminal outcome as the primary reward signal rather than action imitation.

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

- **POSITIVE** — `org_name` contains an organism name (no negative keyword present); or `comments` contains colony counts, growth language, contamination, or other clinical indicators of a positive result
- **NEGATIVE** — `org_name` or `comments` contains language indicating no organism detected: no growth, not detected, negative, nonreactive, indeterminate, or no [organism] seen/found/isolated
- **CANCELLED** — `org_name` or `comments` indicates the test was not completed: cancelled, test not performed, or patient credited
- **OTHER** — both `org_name` and `comments` are null, or `comments` contains only placeholder characters (underscores, dashes) with no interpretable content

`org_name` alone is not a fully reliable indicator — some positives are documented only in free-text `comments`. The classifier handles both columns to maximize coverage without manual review.

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
| Current vitals | `current_temperature`, `current_heartrate`, `current_resprate`, `current_o2sat`, `current_sbp`, `current_dbp`, `current_pain`, `current_pulse_pressure`, `current_map` |
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

# Actions

## Microbiology / Culture Orders

Microbiology cultures are represented as discrete actions by specimen type. The `action_space` column takes the `spec_type_desc` value for the top 20 most frequent specimen types and buckets all remaining specimen types into `'OTHER'`.

### Rationale
The top 20 specimen types cover ~97% of all microbiology records in the cohort. The remaining types are rare, heterogeneous, and do not represent clinically distinct decisions in the ED context — bucketing them as `OTHER` preserves signal from the long tail without inflating the action space. The specific specimen type ordered (blood culture vs. urine vs. swab, etc.) carries meaningful clinical signal about what infection the clinician suspects, so collapsing all cultures into a single binary action would discard this information.

Records where `storetime > stay_window_end` (result came back after the patient's care window ended) are dropped. Stay window end is: `first_icu_intime` for ICU patients, `dischtime` for admitted non-ICU patients, and `ed_outtime` for ED-only discharges. Results arriving after the stay cannot be used as real-time state features and cannot inform the `culture_positive` retrospective label for the relevant care episode.

## Medication Actions

Medication administration is represented as **4 discrete actions**. For admitted patients, these are derived from the `event_txt` column of the `hosp.emar` table. For ED-only patients (and ED-phase events for admitted patients), Pyxis dispenses from `ed.pyxis` map to `administer_medication` — the only action available during an ED stay, since drips, rate changes, and stops are inpatient concepts.

| Action | emar `event_txt` values mapped | Clinical meaning |
|---|---|---|
| `administer_medication` | Administered, Partial Administered, Delayed Administered, Administered Bolus from IV Drip, Administered in Other Location | One-time dose (pill, injection, bolus, IV push) |
| `start_medication` | Started, Restarted | Begin continuous infusion or drip. Restart is collapsed here — the agent's decision is identical to starting; it is a documentation artifact of a prior stop. |
| `stop_medication` | Stopped, Stopped As Directed, Stopped - Unscheduled, Stopped in Other Location | Discontinue an active continuous infusion or drip |
| `rate_change` | Rate Change | Titrate dose of an active drip without stopping it |
| *(excluded)* | Flushed, Not Flushed, Flushed in Other Location | IV line maintenance — not a medication decision |
| *(excluded)* | Applied, Removed, Removed Existing / Applied New, Not Removed | Topical/patch administration — different care context, excluded for simplicity |
| *(excluded)* | Assessed, Not Assessed, Confirmed, Not Confirmed | Documentation and verification events — not clinical actions |
| *(excluded)* | Not Given, Not Applied, Not Given per Sliding Scale | Non-events — medication was not administered; documentation artifacts |
| *(excluded)* | Hold Dose | Intentional skip of a scheduled dose — not the same as stopping; excluded to avoid ambiguity with `stop_medication` |
| *(excluded)* | Infusion Reconciliation, in Other Location | Administrative reconciliation events |
| *(excluded)* | Delayed, Not Started, Not Stopped | Incomplete or cancelled intentions |

### Rationale

The `administer_medication` / `start_medication` distinction captures the clinically important difference between a one-time dose and the initiation of a continuous drip. Drip management (start, stop, titrate) is particularly relevant for ICU escalation decisions — vasopressor initiation and titration are strong signals of patient acuity.

`rate_change` is retained as a separate action rather than folded into `start_medication` because titration frequency is itself a meaningful signal: patients requiring repeated upward titrations are deteriorating, while downward titration may precede successful discharge. Merging these would obscure that pattern.

### Medication State Representation

The active medication state (what drugs are currently running) is represented separately in the state vector, not through the action space. This handles the case where multiple medications are active simultaneously — each drug or drug class has its own state flag rather than being encoded as a compound action. The specific design of the medication state features is deferred to the feature engineering phase (Stage 2.1) and will be documented here once finalized.


## Lab Orders

Labs are represented as **19 discrete actions**, one per unique combination of `category` and `fluid` present in the ED cohort. This replaces the prior design of 35 individual actions per unique `label`.

### Rationale for Category × Fluid Action Space

The MIMIC-IV lab data includes a `category` field (e.g., "Hematology", "Chemistry") and a `fluid` field (e.g., "Blood", "Urine") that together describe the type of lab ordered at a higher level of abstraction than the individual `label`. Grouping by these two fields yields 19 unique combinations, reducing the action space from 35 to 19 while still capturing the clinically meaningful distinctions between different lab types (e.g., blood chemistry vs. urine chemistry are genuinely different ordering decisions).

**Limitation:** Collapsing to category × fluid means the agent cannot learn to target a single specific test within a group — it acts on the combination as a whole. In practice, individual tests within the same category × fluid group tend to be ordered together, so this is an acceptable tradeoff for a more tractable action space.

### Lab State Representation

**Original design:** Each category × fluid combination was initially encoded as a single 3-state ordinal feature: `0` = not ordered, `1` = result normal, `2` = result abnormal. Worst-case aggregation governed the value within each group (any abnormal result set the feature to 2).

**Current design:** Each category × fluid combination is encoded as three separate binary OHE columns:
- `{category}-{fluid}_Normal` -- results have returned and all are within normal limits
- `{category}-{fluid}_Pending` -- an order exists but no result has been returned yet
- `{category}-{fluid}_Abnormal` -- at least one result in the group is flagged as abnormal

**Rationale for change from ordinal to OHE:**
1. **Model performance:** OHE produced better results in practice by allowing the model to learn independent weights for each status rather than assuming a single linear relationship across the ordinal scale.
2. **Label distance:** Ordinal encoding implies equal semantic distances between adjacent values (0→1 = 1→2). This assumption does not hold -- "not ordered" to "pending" is not the same clinical distance as "normal" to "abnormal." OHE eliminates this false distance assumption.

**Grouping logic (unchanged):** When multiple individual test results within the same category × fluid combination are available at the same time step, worst-case aggregation still governs the Abnormal flag -- if any result is flagged, `_Abnormal = 1`. If all results are normal, `_Normal = 1`. If an order exists with no result yet, `_Pending = 1`.

**Most recent result update:** At each time step the OHE columns reflect only the most recent status for that combination -- prior results are not accumulated. This keeps the state vector fixed-size and ensures the agent is always acting on the current best information.

# Outlier Handling for Stay Length

Stays in the extreme tail of the length-of-stay distribution will be removed from the training cohort. Exceptionally long hospital stays introduce non-representative trajectories with unusual lab ordering patterns driven by chronic care rather than acute decision-making, which can distort the learned policy. A length-of-stay cutoff (e.g., 95th or 99th percentile) will be applied during cohort construction; the exact threshold will be determined during EDA.

# Cohort DF EDA/Prep

## Triage Outliers/Missing

Triage vitals in the MIMIC-IV ED dataset contain a significant number of implausible values, likely due to data entry errors (extra digits, wrong units, transposed values). The strategy is to correct values where a clear pattern exists, null out values where no reliable correction is possible, and impute remaining nulls using KNN in a later step. Corrections are applied in a fixed order within each variable because some transforms create values that are then caught by a downstream rule.

### Temperature

Normal range: 97–99°F. Hypothermia below 95°F, severe below 82°F. Hyperthermia above 100°F, severe above 104°F. Recorded values extend up to 110–115°F in rare cases.

| Rule | Action | Assumption |
|---|---|---|
| Value > 900 | Divide by 10 | Extra digit entered (e.g. 986 → 98.6) |
| 28 < value ≤ 40 | Convert Celsius → Fahrenheit: `(x * 1.8) + 32` | Value was recorded in Celsius instead of Fahrenheit |
| 5 < value < 10 | Multiply by 10 | Missing leading digit (e.g. 9.8 → 98) |
| Value > 115 | Set to null | No recoverable correction pattern; KNN imputation |

Values below 82°F that survive all transforms (e.g. the 46–80°F range) are left as-is and will be addressed in a downstream validation step or absorbed by KNN imputation. There is not enough confidence to correct or null these without additional context.

### Heart Rate

Normal range: 60–100 bpm. Tachycardia above 100, bradycardia below 60. Values below 20 are inconsistent with a living patient at triage.

| Rule | Action | Assumption |
|---|---|---|
| Value > 500 | Divide by 10 | Extra digits entered (e.g. 800 → 80) |
| Value < 20 | Set to null | No clear correction pattern; KNN imputation |

Values in the 256–500 range are left as-is. Spot-checking suggests most are plausible extreme tachycardia, and there is no consistent correction pattern for the minority that may be errors.

### Respiratory Rate

Normal range: 12–20 breaths/min. Above 20–25 can indicate tachypnea.

| Rule | Action | Assumption |
|---|---|---|
| Value > 1000 | Divide by 100, round | Two extra digits entered (e.g. 1800 → 18) |
| 100 < value ≤ 1000 | Divide by 10, round | One extra digit entered (e.g. 180 → 18) |
| Value < 4 | Set to null | No clear correction pattern; KNN imputation |

The divide-by-100 rule is applied before divide-by-10 to avoid double-correcting values above 1000.

### O2 Saturation

Normal range: 95–100%. Clinically significant drop below 88%.

| Rule | Action | Assumption |
|---|---|---|
| 900 < value < 1010 | Divide by 10, floor | Extra digit entered (e.g. 980 → 98, 1000 → 100) |
| 0 < value ≤ 10 | Add 90 | Missing leading digit (e.g. 8 → 98) |
| value == 0, value > 100, or value < 40 | Set to null | Not recoverable; KNN imputation |

Values in the 40–88 range are left as-is — many in this range are clinically plausible (severely hypoxic patients), and there is not enough confidence to null them without individual review.

### Systolic Blood Pressure

Normal range: 90–120 mmHg. Elevated 120–140, hypertensive 140+. Critically low below 50–60.

| Rule | Action | Assumption |
|---|---|---|
| Value > 270 | Set to null | Spot-checking found no consistent correction pattern above this threshold |
| Value < 40 | Set to null | Too low to be plausible at triage |

Values between 40 and 270 are retained. There is a cluster near 24–40 that is suspicious but enough values in that range appear accurate upon spot-checking to avoid a blanket null.

### Diastolic Blood Pressure

Normal range: 60–80 mmHg. Elevated 80–120+. Values below 50 start to get low.

| Rule | Action | Assumption |
|---|---|---|
| Value > 150 | Set to null | Likely charting errors with no clear correction pattern |
| Value < 20 | Set to null | Too low to be plausible |

## Stay Window Correction (Inverted Times)

A small number of records had `stay_window_end` before `stay_window_start`, producing negative `time_steps`. The common pattern among these records was in-hospital death — discharge times were recorded in a disjointed way relative to the ED arrival, causing the window to appear inverted.

**Fix:** For each affected `hadm_id`, the true min `intime` and max `outtime` were looked up directly from the ICU stay records and used to replace the incorrect `stay_window_start` and `stay_window_end` values. These corrected times were saved to `df_grouped.csv` and applied via a map on `hadm_id` before `time_steps` is calculated.

A small number of records (~5) had no hospital stay at all (null stay window after correction) and were dropped from the cohort.

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

**Rationale for Unknown/Declined → Other:** These patients have a race — we simply don't know what it is. Creating a separate `Unknown` category would imply a meaningful distinction that doesn't exist. Collapsing them into `Other` is more honest about what the data actually represents.

## Pain

The `pain` column is a 0–10 self-reported pain scale recorded at triage. It is not used as a triage vital for imputation purposes but is retained as a state feature. Missing or non-numeric values are masked as `'Other'` in the patient state rather than imputed, since pain is subjective and patient-reported — imputing it from other vitals would not be meaningful.

**Main issues:**
- Mixed types: the column contains integers, floats, free-text strings, and punctuation artifacts (quotes, `>`, `-`, `+`)
- Range entries: some values are entered as a range (e.g. `5-7`) rather than a single number
- Values above 10: present in significant volume; could be mis-entries, carryovers from another field, or a different scale (e.g. Glasgow Coma Scale, which would partially explain the large cluster at 13)
- Float values: the scale is 0–10 integers; floats need to be rounded

| Step | Action | Assumption |
|---|---|---|
| Strip punctuation and normalize case | Remove `"`, `"`, `"`, `>`, `-`, `+` from start/end; lowercase | These are data entry artifacts, not meaningful values |
| Range entries `#-#` | Replace with first (lower) number | The lower end of a reported range is the more conservative estimate |
| Coerce to numeric | Non-numeric strings → NaN | Free-text entries are not recoverable as a pain score |
| Values > 10 | Set to NaN | Not a valid 0–10 scale entry; source is ambiguous and not correctable |
| Float values | Round to nearest integer | Pain scale is whole numbers only |
| Remaining NaN | Fill with `'Other'` | Missing pain scores are masked in the patient state rather than imputed |

# Vitals EDA/Prep

## Rhythm Column

The `rhythm` column (cardiac rhythm, e.g. "Sinus Rhythm", "Atrial Fibrillation") is dropped from the vitals dataframe. It is too sparse and too high-cardinality to impute meaningfully, and encoding it as a categorical state feature is deferred pending a decision on how to represent rhythm in the action/state space.

## Triage Row Injection

Before imputation, the triage vitals from the cohort dataframe (`cohort_with_triage`) are injected as a synthetic first reading for each ED stay. The triage row uses `ed_intime` as its `charttime` and carries the triage values for all numeric vital columns. This serves two purposes:
1. Anchors the time series at the start of the stay so ffill/bfill has a starting value to propagate
2. Ensures patients who have triage vitals but no charted vitals readings still contribute imputable data

## Missing Value Imputation Strategy

Imputation is applied in three passes in order:

1. **Forward fill within stay** — sorts by `(ed_stay_id, charttime)` then propagates the last known reading forward. The last recorded value is the best real-time estimate of the current patient state.
2. **Backward fill within stay** — fills leading NaNs at the start of a stay (where ffill has nothing to propagate from). Both passes are bounded to the stay — values never bleed across stays.
3. **IterativeImputer** — for stays that had no reading at all for a given column after ff/bfill, `sklearn.impute.IterativeImputer` models each feature as a function of the other vital columns to fill remaining nulls. `random_state=10`.

Pain and rhythm are excluded from this imputation pipeline. Pain `'Other'` values (non-numeric entries) are retained as-is — see Pain section above.

## Feature Engineering

After imputation, the following features are derived from the numeric vital columns (`temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`):

### Time-based features (per stay, sorted by charttime)

| Feature | Formula | Notes |
|---|---|---|
| `time_since_last_hrs` | `diff(charttime)` in hours | Hours since previous reading within the stay. First reading of each stay is NaN. Also a feature in its own right — frequent readings signal closer monitoring. |
| `{col}_rolling1h` | 1-hour trailing rolling mean | Time-based window (not row-based) so irregular sampling is handled correctly. Computed per stay using `set_index('charttime').rolling('1h')`. |
| `{col}_delta` | `diff()` of column values | 1-step change from previous reading. First reading of each stay is NaN. |
| `{col}_rate_per_min` | `{col}_delta / time_since_last_min` | Rate of change per minute. Normalises delta for the time gap between readings. |

### Blood pressure derived features

| Feature | Formula | Clinical meaning |
|---|---|---|
| `pulse_pressure` | `sbp - dbp` | Reflects stroke volume and arterial stiffness. Wide PP can indicate sepsis or aortic regurgitation; narrow PP suggests low cardiac output. |
| `map` | `dbp + (sbp - dbp) / 3` | Mean Arterial Pressure — standard clinical measure of average perfusion pressure. Often more predictive of organ perfusion than sbp or dbp alone. |

# Lab Event EDA/Prep

## Lab Event Simplification

The raw lab events table contains one row per individual lab test result, meaning a single lab order batch (e.g. a CBC) can produce 20+ rows for the same patient at the same timestamp. To simplify the action space and reduce redundancy, the table is collapsed to one row per `(ed_stay_id, category, fluid, order_time)` group.

**Action space:** Instead of using individual lab `label` as the action (high cardinality), we use the unique combination of `category` and `fluid`. This yields 19 possible lab actions the agent can take, which is tractable and clinically meaningful — e.g. ordering Hematology/Blood is a distinct decision from ordering Chemistry/Urine.

**Collapsing logic:**
- One row per `(ed_stay_id, category, fluid, order_time)` — this represents a single lab action at a point in time
- `abnormal`: `True` if any individual result in the group had a non-null flag, `False` otherwise. Non-null flag = abnormal result; null = normal.
- `result_time`: max across the group — represents when all results in the batch are back
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

**Data preparation:** `modeling/data_prep/traditional_ml.py`

**Train/test/validation split:** 80/10/10, split by `subject_id` (not by `ed_stay_id`) to prevent data leakage across multiple visits by the same patient. Stratified on the binary label to preserve class balance across splits.

**Scaling:** `StandardScaler` fit on the training set, applied to test and validation. Scaling columns are defined in `modeling/config/traditional_ml.yaml`.

**Chief complaint:** OHE category encoding for Logistic Regression and Random Forest; TF-IDF (top 50 terms, fit on train only) for XGBoost.

**Class imbalance:** Logistic Regression and Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight` calculated from the training set class ratio (negative count / positive count).

**Scripts:** `modeling/traditional_ml/{logistic_regression,random_forest,xgboost_train}.py`

**Artifacts:** Saved to `modeling/artifacts/{log_reg,random_forest,xgboost}/` -- model pickle, scaler pickle, test and validation parquets.

## LSTM

The LSTM operates on the full event-driven time series from `full_patient_state`, using variable-length sequences per stay. This allows the model to capture temporal patterns in vital sign trends, lab result timing, and treatment sequences that the one-row aggregation discards.

Padding and masking are handled in the model dataloader at training time -- the stored dataset is not padded.

## Offline RL Agent

The RL agent learns a treatment policy from historical clinician behavior using offline RL (no environment interaction). The agent observes the full patient state at each time step and selects from the action space defined in the Actions section. The primary reward signal is terminal outcome (discharge = positive, ICU transfer = negative), with intermediate step rewards under development.

# Rewards

## Estimating values of rewards

The reward function is designed to provide both a terminal signal (end-of-stay outcome) and intermediate per-step shaping signals.

**Terminal reward:**
- Discharge: `+1` -- the patient was successfully managed and released from the ED
- ICU transfer: `-1` -- the patient deteriorated to a level requiring intensive care

The terminal reward reflects the binary classification target and aligns the RL agent's objective with the supervised learning task.

**Intermediate rewards (under development):**
Per-step reward shaping is under active design. Candidate approaches include:
- Small negative reward per step to encourage efficiency (penalize unnecessarily prolonged stays)
- Penalty for ordering redundant or low-yield tests (e.g., re-ordering the same lab when the prior result was normal and no clinical change occurred)
- Penalty for medication actions inconsistent with the patient's current vital state

The specific values and formulas for intermediate rewards will be documented here once finalized through experimentation.