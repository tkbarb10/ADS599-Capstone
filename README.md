# ADS599-Capstone
Capstone project

## data_pipelines

```
data_pipelines/
├── pipeline_scripts/          -- orchestration scripts (BQ load -> preprocess -> feature eng -> HF push)
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
├── combine_patient_state/     -- builds full event-driven patient state DataFrame and pushes to HF
│   ├── main.py                -- entry point: checkpoint/resume, calls all steps, pushes to HF
│   ├── preprocessing/
│   │   ├── cohort.py          -- load_cohort()
│   │   └── step_index.py      -- collect_event_times(), build_step_index(), drop_out_of_window(),
│   │                             drop_single_step_stays()
│   ├── feature_engineering/
│   │   ├── ohe_utils.py       -- merge_pivot_into_patient() [shared by labs/micro/meds]
│   │   ├── vitals.py          -- snap_vitals(), compute_time_since_last_vitals()
│   │   ├── labs_ohe.py        -- expand_labs_ohe()
│   │   ├── micro_ohe.py       -- expand_micro_ohe()
│   │   ├── meds_flags.py      -- expand_med_flags()
│   │   ├── imaging.py         -- snap_ecg_ohe(), snap_rad_ohe()
│   │   └── patient_features.py -- add_location_flags(), add_height_weight(),
│   │                              add_medrecon(), add_derived_features()
│   └── validation_checks/
│       └── checks.py          -- run_all_checks()
├── preprocessing_scripts/     -- data cleaning functions imported by pipeline scripts
│   ├── admitted_meds_preprocessing.py
│   └── cohort_preprocessing.py
└── feature_engineering/       -- feature construction functions imported by pipeline scripts
    ├── cohort_df_features.py
    ├── ecg_features.py
    └── meds_features.py
```

Each pipeline script follows the same pattern:
1. Load raw data from BigQuery via `utils/bq.py`
2. Clean with functions from `preprocessing_scripts/`
3. Engineer features with functions from `feature_engineering/`
4. Push processed dataset to HuggingFace via `datasets.push_to_hub`

Run the full patient state pipeline:
```
python -m data_pipelines.combine_patient_state.main
```

HuggingFace repo targets and split/config names are centrally managed in `project_setup/settings.yaml`.