# ADS599 Capstone -- ED Clinical Decision Support

A research prototype for predicting ICU transfer risk in emergency department patients using MIMIC-IV EHR data. The system combines a sequential LSTM classifier with an offline reinforcement learning agent to simulate how a clinical decision support tool could recommend ICU transfer co, discharge or continued observation in real time.

---

## Purpose

Emergency departments face a core challenge: identifying which patients will deteriorate and require ICU-level care before the clinical picture becomes obvious. This project trains models on historical ED event sequences from MIMIC-IV to:

1. Predict step-by-step ICU transfer probability as clinical events unfold (LSTM)
2. Learn a policy for when to recommend ICU transfer or discharge vs. continued observation (CQL reinforcement learning agent)
3. Provide a prototype clinical decision support dashboard for exploring model behavior on individual patients

---

## Main Findings

- The LSTM classifier achieves strong discrimination on the held-out test set (PR-AUC 0.977, ROC-AUC 0.994, Brier Score 0.011), substantially outperforming all traditional ML baselines (XGBoost PR-AUC 0.606, Random Forest 0.557, Logistic Regression 0.521).
- The primary driver of performance differences across patient subgroups is local class imbalance (ICU rate within length-of-stay bins), not model behavior or stay duration.
- The RL agent commits to a decision on 46.3% of test stays. Among those, it recommends earlier than the treating provider in 24.4% of committed stays (median 4 events earlier). Macro F1 at the point of first commit is 0.8531; at the terminal event it is 0.9711.
- The reward function has a structural limitation in offline RL: penalties for wrong decisions are unreachable at terminal rows because the provider action is the ground truth label by definition. A redesigned per-step penalty weighted by LSTM confidence is recommended for future work.
- SHAP attributions are directionally interpretable but should be treated as hypotheses -- the model may attend to proxy signals (activity volume, missingness patterns) rather than directly clinically meaningful physiology.

---

## Data Access

This project uses **MIMIC-IV** (Medical Information Mart for Intensive Care IV), a de-identified EHR database from Beth Israel Deaconess Medical Center. Access requires:

1. Complete the CITI "Data or Specimens Only Research" course
2. Sign the PhysioNet credentialed data use agreement at [physionet.org](https://physionet.org/content/mimiciv/3.1/)

Processed datasets used for modeling are hosted on HuggingFace (private) and loaded via the `datasets` library. Raw data is pulled from BigQuery using a Google Cloud project with PhysioNet access.

---

## Project Structure

```
ADS599-Capstone/
├── project_setup/
│   ├── cohort_base.py             -- builds base cohort in BigQuery; run first
│   └── settings.yaml              -- HuggingFace repo targets, split/config names, logging paths
│
├── data_pipelines/
│   ├── pipeline_scripts/          -- BigQuery extract -> clean -> feature engineer -> HF push
│   │   ├── cohort_pipeline.py     -- run second; loads cohort from BQ, pushes to HF
│   │   ├── triage_vitals_pipeline.py  -- run before other feature pipelines
│   │   ├── labs_pipeline.py
│   │   ├── microbiology_pipeline.py
│   │   ├── dispensed_meds_pipeline.py
│   │   ├── medrecon_pipeline.py
│   │   ├── radiology_data.py
│   │   ├── ecg_data.py
│   │   └── omr_pipeline.py
│   ├── combine_patient_state/     -- builds the full event-driven patient state DataFrame
│   │   ├── main.py                -- entry point; checkpoint/resume; calls all steps
│   │   ├── preprocessing/         -- cohort loading, step index construction
│   │   ├── feature_engineering/   -- vitals, labs OHE, micro OHE, meds flags, imaging, patient features
│   │   └── validation_checks/
│   ├── preprocessing_scripts/     -- cleaning functions (cohort, meds, radiology)
│   └── feature_engineering/       -- feature construction functions (cohort, ECG, meds)
│
├── modeling/
│   ├── data_prep/                 -- shared data loading, splitting, scaling for all models
│   │   ├── traditional_ml.py      -- first-hour aggregation, CC encoding, train/test/val split
│   │   └── lstm.py                -- full sequence loading, padding, DataLoader construction
│   ├── traditional_ml/            -- logistic regression, random forest, XGBoost training and tuning
│   ├── lstm/                      -- LSTM model definition, training, step-by-step inference, SHAP
│   └── rl_agent/                  -- CQL agent, environment, training, evaluation
│
├── docs/
│   ├── evaluation.md              -- all model results and metrics
│   ├── decisions.md               -- data and modeling decisions log
│   └── rl_iteration.md            -- RL agent iteration history
│
├── eda/                           -- exploratory analysis notebooks
├── utils/                         -- shared helpers (BigQuery, YAML loading, logging)
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Google Cloud project with access to `physionet-data` BigQuery datasets
- HuggingFace account with access to the private modeling data repo

### Install

```bash
git clone https://github.com/<your-org>/ADS599-Capstone.git
cd ADS599-Capstone
uv sync          # or: pip install -e .
```

### Environment

Create a `.env` file at the project root with your credentials:

```
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
HF_TOKEN=hf_...
PROJECT_NAME=your-gcp-project-id
```

---

## What Needs to Run First

### Step 1 -- Build the base cohort in BigQuery

```bash
python -m project_setup.cohort_base
```

This must run before anything else. It builds the base patient cohort table in BigQuery that all downstream feature queries reference.

### Step 2 -- Push cohort to HuggingFace

```bash
python -m data_pipelines.pipeline_scripts.cohort_pipeline
```

### Step 3 -- Feature pipelines

Run triage/vitals first (other pipelines depend on its output), then the rest in any order:

```bash
python -m data_pipelines.pipeline_scripts.triage_vitals_pipeline

python -m data_pipelines.pipeline_scripts.labs_pipeline
python -m data_pipelines.pipeline_scripts.microbiology_pipeline
python -m data_pipelines.pipeline_scripts.dispensed_meds_pipeline
python -m data_pipelines.pipeline_scripts.medrecon_pipeline
python -m data_pipelines.pipeline_scripts.radiology_data
python -m data_pipelines.pipeline_scripts.ecg_data
python -m data_pipelines.pipeline_scripts.omr_pipeline
```

### Step 4 -- Combine into full patient state

```bash
python -m data_pipelines.combine_patient_state.main
```

This builds the event-driven patient state DataFrame and pushes it to HuggingFace. Supports checkpoint/resume if interrupted.

### Step 5 -- Train models

```bash
# Traditional ML
python -m modeling.traditional_ml.logistic_regression
python -m modeling.traditional_ml.random_forest
python -m modeling.traditional_ml.xgboost_train

# LSTM (GPU recommended; see modeling/lstm/colab_kernel_access.ipynb for Colab setup)
python -m modeling.lstm.train

# Step-by-step LSTM inference (required before RL training)
python -m modeling.lstm.step_by_step_train

# RL agent
python -m modeling.rl_agent.train
```

### Step 6 -- Evaluate

```bash
python -m modeling.traditional_ml.evaluate
python -m modeling.lstm.evaluate
python -m modeling.rl_agent.evaluate
```

All HuggingFace repo targets and split/config names are centrally managed in `project_setup/settings.yaml`.

---

## Suggestions for Future Work

- **Reward function redesign** -- apply per-step penalties weighted by LSTM confidence (`p_icu`) on non-terminal rows so wrong-decision penalties are reachable during training
- **POMDP framing** -- the current MDP assumes full state observability; modeling the belief state over true patient condition would better reflect clinical uncertainty
- **Prospective validation** -- model performance was evaluated on a held-out historical split; prospective validation on a separate institution or time period is needed before any clinical consideration
- **Richer action space** -- the current agent chooses between `wait`, `discharge`, and `transfer_icu`; adding intermediate actions (escalate monitoring, order specific labs) would make the policy more clinically actionable
- **Online learning** -- offline RL is limited by distributional shift; a framework that allows the agent to update from new clinical outcomes over time would improve generalization
- **Explainability** -- replace GradientExplainer with DeepLIFT or exact Shapley methods to get complete, additive attributions; GradientExplainer is an approximation that does not satisfy completeness

---

## Limitations

- MIMIC-IV is drawn from a single academic medical center (Beth Israel Deaconess Medical Center, Boston); generalizability to other institutions is unknown
- The class imbalance (~7.9% ICU transfer rate) creates evaluation challenges; PR-AUC is the primary metric
- The RL agent is trained offline on historical provider decisions; it cannot explore counterfactual actions and may inherit provider biases
- SHAP attributions reflect model attention, not validated clinical causality

---

## Live Demo

The prototype dashboard is hosted at:
[https://huggingface.co/spaces/ADS599-Capstone/Clinical_Support_Decision_Tool](https://huggingface.co/spaces/ADS599-Capstone/Clinical_Support_Decision_Tool)

---

## References

- Johnson, A., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T., Hao, S., Moody, B., Mark, R., Osta-Schlicht, C., & Celi, L.A. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1), 1. https://doi.org/10.1038/s41597-022-01899-x
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS 2020*.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. M., & Lee, S.-I. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence*.

---

## Contact

**Taylor Kirk** -- [tkirk@sandiego.edu](mailto:tkirk@sandiego.edu)

**April Chia** -- [achia@sandiego.edu](mailto:achia@sandiego.edu)

**Tommy Barron** -- [tbarron@sandiego.edu](mailto:tbarron@sandiego.edu)
