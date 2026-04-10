# ADS599-Capstone
Capstone project

## data_pipelines

```
data_pipelines/
├── pipeline_scripts/          — orchestration scripts (BQ load → preprocess → feature eng → HF push)
│   ├── cohort_pipeline.py
│   ├── dispensed_meds_pipeline.py
│   ├── ecg_data.py
│   ├── labs_pipeline.py
│   ├── medrecon_pipeline.py
│   ├── microbiology_pipeline.py
│   ├── omr_pipeline.py
│   ├── patient_state_df.py
│   ├── radiology_data.py
│   ├── triage_pipeline.py
│   └── vitals_pipeline.py
├── preprocessing_scripts/     — data cleaning functions imported by pipeline scripts
│   ├── admitted_meds_preprocessing.py
│   └── cohort_preprocessing.py
└── feature_engineering/       — feature construction functions imported by pipeline scripts
    ├── cohort_df_features.py
    ├── ecg_features.py
    └── meds_features.py
```

Each pipeline script follows the same pattern:
1. Load raw data from BigQuery via `utils/bq.py`
2. Clean with functions from `preprocessing_scripts/`
3. Engineer features with functions from `feature_engineering/`
4. Push processed dataset to HuggingFace via `datasets.push_to_hub`

HuggingFace repo targets and split/config names are centrally managed in `project_setup/settings.yaml`.