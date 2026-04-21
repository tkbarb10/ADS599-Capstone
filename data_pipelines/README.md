# data_pipelines/

Orchestrates all data extraction, cleaning, feature engineering, and HuggingFace upload steps. Each pipeline script loads a SQL query from `sql_scripts/`, processes the result, and pushes a cleaned dataset to the private HuggingFace repo defined in `project_setup/settings.yaml`.

---

## Directory Layout

```
data_pipelines/
├── pipeline_scripts/          -- entry points; one script per feature domain
├── combine_patient_state/     -- assembles the full event-driven patient state DataFrame
├── preprocessing_scripts/     -- cleaning functions imported by pipeline scripts
└── feature_engineering/       -- feature construction functions imported by pipeline scripts
```

---

## pipeline_scripts/

Each script follows the same pattern: load from BigQuery -> clean -> engineer features -> push to HuggingFace.

| Script | Output dataset (HuggingFace) | Run order |
|---|---|---|
| `cohort_pipeline.py` | `cohort` | 1 -- after `project_setup/cohort_base.py` |
| `triage_vitals_pipeline.py` | `triage_vitals` | 2 -- other pipelines depend on this |
| `labs_pipeline.py` | `labs` | 3+ (any order) |
| `microbiology_pipeline.py` | `microbiology` | 3+ |
| `dispensed_meds_pipeline.py` | `dispensed_meds` | 3+ |
| `medrecon_pipeline.py` | `medrecon` | 3+ |
| `radiology_data.py` | `radiology` | 3+ |
| `ecg_data.py` | `ecg` | 3+ |
| `omr_pipeline.py` | `omr_height`, `omr_weight` | 3+ |

**Logs** -- all pipeline scripts write to `validation.log` at the project root (path configured in `project_setup/settings.yaml` under `logging.path`).

---

## combine_patient_state/

Joins all feature datasets into a single event-driven patient state DataFrame and pushes it to HuggingFace as `full_patient_state`. This is the input for all modeling pipelines.

**Entry point:**
```bash
python -m data_pipelines.combine_patient_state.main
```

**Processing steps (in order):**

1. Load cohort -- provides labels and stay window boundaries
2. Build step index -- collects all event timestamps per stay, assigns a sequential `step_idx`
3. Snap vitals -- forward-fills vital signs onto each step
4. Labs OHE -- expands lab results into one-hot encoded columns per test
5. Microbiology OHE -- expands culture results into binary flags
6. Dispensed medications -- merges med flags onto each step
7. ECG and radiology -- snaps imaging presence flags per step
8. Location flags, height/weight, medrecon -- static and stay-level features
9. Derived features -- arrival transport OHE, acuity forward-fill, total stay length

**Checkpoint/resume** -- intermediate state is saved to a local `.parquet` checkpoint file after each step. If the script is interrupted, re-running it will resume from the last completed step rather than starting over. The checkpoint is deleted automatically on successful completion.

**Logs** -- writes to `patient_combinations.log` at the project root (configured under `logging.patient_combination_path`).

**Output** -- pushed to the `full_patient_state` config on HuggingFace. Shape at time of writing: ~7M rows x 236+ columns.

---

## preprocessing_scripts/

Cleaning functions imported by `pipeline_scripts/`. Not run directly.

| File | Purpose |
|---|---|
| `cohort_preprocessing.py` | Resolves label conflicts (e.g. discharge label but patient was admitted) |
| `admitted_meds_preprocessing.py` | Filters out non-administration med events (e.g. orders, cancellations) |
| `microbiology_preprocessing.py` | Cleans organism and test name fields |
| `radiology_preprocessing.py` | Filters out administrative/non-diagnostic radiology records |

---

## feature_engineering/

Feature construction functions imported by `pipeline_scripts/`. Not run directly.

| File | Purpose |
|---|---|
| `cohort_df_features.py` | Adds derived cohort-level fields (e.g. boarding hours, ICU flag) |
| `triage_vitals_features.py` | Vital sign deltas, rolling averages, MAP, pulse pressure; forward-fill and missing indicators |
| `lab_features.py` | Lab result normalization and flagging |
| `meds_features.py` | Drug class encoding for dispensed medications |
| `medrecon_features.py` | Pre-arrival med class pivot |
| `ecg_features.py` | ECG feature extraction from machine measurements |
| `drug_classes.py` | Drug class lookup table |
| `radiology_features.py` | Radiology flag construction |
