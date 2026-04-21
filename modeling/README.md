# modeling/

All model training, tuning, evaluation, and inference code. Each model family has its own subdirectory. Shared data loading and preprocessing lives in `data_prep/`.

All inputs are loaded from HuggingFace (`full_patient_state` or downstream datasets). All trained model artifacts are saved locally under `modeling/artifacts/`.

---

## Directory Layout

```
modeling/
├── data_prep/          -- shared data loading, splitting, and scaling
├── traditional_ml/     -- logistic regression, random forest, XGBoost
├── lstm/               -- LSTM classifier and step-by-step inference
├── rl_agent/           -- CQL reinforcement learning agent
├── config/             -- YAML config files for each model family
└── artifacts/          -- saved models, predictions, evaluation outputs (gitignored)
```

---

## data_prep/

Shared preprocessing imported by all training scripts. Not run directly.

| File | Purpose |
|---|---|
| `traditional_ml.py` | Loads `full_patient_state`, filters to first 60 minutes per stay, aggregates to one row per stay, encodes chief complaint (OHE or TF-IDF), stratified train/test/val split, StandardScaler |
| `lstm.py` | Loads full event sequences, pads to max length, constructs PyTorch DataLoaders |
| `rl.py` | Loads `sbs_predictions` (LSTM step-by-step output), builds train/test split for RL agent |
| `columns.py` | Column group definitions shared across data prep modules |

---

## traditional_ml/

First-hour snapshot models. Each model trains on one aggregated row per ED stay.

| Script | What it does | Artifacts saved |
|---|---|---|
| `logistic_regression.py` | Trains logistic regression with balanced class weights | `modeling/artifacts/log_reg/` -- model `.pkl`, test predictions `.parquet` |
| `random_forest.py` | Trains random forest with balanced class weights | `modeling/artifacts/random_forest/` -- model `.pkl`, test predictions `.parquet` |
| `xgboost_train.py` | Trains XGBoost with TF-IDF chief complaint encoding | `modeling/artifacts/xgboost/` -- model `.pkl`, TF-IDF vectorizer `.pkl`, test predictions `.parquet` |
| `tune.py` | Hyperparameter search (RandomizedSearchCV) for all three models | `modeling/artifacts/<model>/` -- best params `.json` |
| `evaluate.py` | Computes metrics and plots for all three models on test set | `modeling/artifacts/evaluation/` -- plots `.png`, metrics `.json` |
| `predict.py` | Runs inference on new data using a saved model | stdout / caller-specified output |

**Logs** -- training writes to `train.log`; tuning to `tune.log`; evaluation to `evaluate.log` (paths in `project_setup/settings.yaml`).

---

## lstm/

Sequential model that processes the full event timeline of each ED stay.

| Script | What it does | Artifacts saved |
|---|---|---|
| `train.py` | Trains the 2-layer LSTM (hidden size 256, 236 input features) | `modeling/artifacts/lstm/` -- model checkpoint `.pt`, training history |
| `tune.py` | Hyperparameter search over LSTM architecture/training params | `modeling/artifacts/lstm/` -- best params `.json` |
| `step_by_step_train.py` | Runs LSTM inference across all events for 40% held-out stays; saves step-by-step probabilities | HuggingFace -- `sbs_predictions` dataset |
| `evaluate.py` | Computes test set metrics, calibration curves, probability distributions, stratified analysis | `modeling/artifacts/lstm/` -- plots `.png`, metrics `.json` |
| `shap.py` | Computes GradientExplainer SHAP values for a patient sample | `modeling/artifacts/lstm/` -- SHAP values `.npy` |
| `predict.py` | Runs inference on a single stay or batch | stdout / caller-specified output |
| `model.py` | LSTM model class definition (imported by training scripts) | -- |
| `training_eval_loops.py` | Train/eval loop functions (imported by `train.py`) | -- |

**Logs** -- training writes to `train.log`; evaluation to `evaluate.log`.

**Note:** GPU is strongly recommended for training. A Colab setup notebook is available at `modeling/lstm/colab_kernel_access.ipynb`.

---

## rl_agent/

Conservative Q-Learning (CQL) agent that learns a transfer/discharge/wait policy from the LSTM step-by-step predictions.

**Prerequisite:** `modeling/lstm/step_by_step_train.py` must run first to generate the `sbs_predictions` HuggingFace dataset.

| Script | What it does | Artifacts saved |
|---|---|---|
| `train.py` | Trains the CQL distributional Q-network | `modeling/artifacts/rl_policy/` -- policy checkpoint `.pt`, training history |
| `tune.py` | Hyperparameter search for the RL agent | `modeling/artifacts/rl_policy/` -- best params `.json` |
| `evaluate.py` | Evaluates policy on test stays: terminal row metrics, first-commit metrics, lead time analysis | `modeling/artifacts/rl_policy/` -- metrics `.json`, plots `.png` |
| `agent.py` | Agent class and action selection logic (imported by training scripts) | -- |
| `environment.py` | RL environment definition: state transitions, reward function (imported by training scripts) | -- |
| `rl_functions.py` | Shared RL utility functions | -- |

**Logs** -- training writes to `rl_train.log`.

---

## config/

YAML configuration files. Hyperparameters, artifact directory paths, and HuggingFace dataset references are all set here -- not hardcoded in training scripts.

| File | Used by |
|---|---|
| `traditional_ml.yaml` | `modeling/traditional_ml/` |
| `lstm.yaml` | `modeling/lstm/` |
| `rl_agent.yaml` | `modeling/rl_agent/` |

---

## artifacts/

All trained models, predictions, and evaluation outputs are written here. This directory is gitignored -- nothing under `artifacts/` is committed to the repo.

```
modeling/artifacts/
├── log_reg/           -- logistic regression model + predictions
├── random_forest/     -- random forest model + predictions
├── xgboost/           -- XGBoost model + TF-IDF vectorizer + predictions
├── lstm/              -- LSTM checkpoint + SHAP values + eval outputs
├── step_by_step/      -- intermediate step-by-step inference outputs
├── rl_policy/         -- RL policy checkpoint + eval outputs
└── evaluation/        -- cross-model evaluation plots and metrics
```
