"""
modeling/lstm/step_by_step_train.py

Trains an LSTM sequence classifier with a smaller training split, then runs
step-by-step inference on the combined val+test set to produce per-timestep
ICU transfer probabilities for use as the RL agent reward signal.

Pipeline:
  1. Load and prep data via prep_data (same as train.py, adjustable train_size)
  2. Train with early stopping on val loss
  3. Wrap trained model in StepByStepWrapper
  4. Run inference on combined val+test DataLoader
  5. Save DataFrame with p_icu column to artifacts/sbs/

Usage:
    python -m modeling.lstm.step_by_step_train
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import f1_score

from modeling.lstm.train import prep_data
from modeling.lstm.model import LSTMSequenceModel, StepByStepWrapper
from modeling.lstm.training_eval_loops import training_loop, evaluation_loop
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging

settings = load_yaml("project_setup/settings.yaml")
cfg = load_yaml("modeling/config/lstm.yaml")
logger = setup_logging(settings["logging"]["train_path"])

PROJECT_ROOT = Path(__file__).parents[2]

# -- Config -------------------------------------------------------------------
random_state = cfg["random_state"]
train_cfg = cfg["training"]
artifact_cfg = cfg["artifacts"]
hf_cfg = settings["hugging_face"]

train_size = cfg['step_by_step']['train_size']
batch_size = cfg['data']['batch_size']
pad_length = cfg['data']['pad_length']

epochs = train_cfg["epochs"]
learning_rate = train_cfg["learning_rate"]
weight_decay = train_cfg["weight_decay"]
optimizer_key = train_cfg["optimizer"]
training_update = train_cfg["training_update"]
patience = train_cfg["patience"]
tol = train_cfg["tol"]

optimizer_options = {
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
}

# -- Reproducibility ----------------------------------------------------------
torch.manual_seed(random_state)
np.random.seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -- Device -------------------------------------------------------------------
device = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else "cpu"
)
logger.info(f"Using device {device}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    pad_train, pad_val, pad_test, scaler, df_test, df_val, class_weight = prep_data(
        hf_cfg=hf_cfg, device=device, train_size=train_size
    )

    # -- Train ----------------------------------------------------------------
    seq_model = LSTMSequenceModel().to(device)
    optimizer = optimizer_options[optimizer_key](
        seq_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    train_losses = []
    eval_losses = []
    best_eval_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}--------------------")
        train_loss = training_loop(
            train_data_loader=pad_train,
            model=seq_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            training_update=training_update,
        )
        train_losses.append(train_loss)

        eval_loss, preds, true_labels = evaluation_loop(
            test_data_loader=pad_val,
            model=seq_model,
            loss_fn=loss_fn,
            device=device,
        )
        eval_losses.append(eval_loss)

        pred_labels = preds.argmax(dim=1)
        f1_both = f1_score(true_labels, pred_labels, average=None)
        f1_macro = f1_score(true_labels, pred_labels, average="macro")
        logger.info(
            f"\nTrain loss: {train_loss:.4f}  Val loss: {eval_loss:.4f}"
            f"\nF1 Transfer: {f1_both[1]:.4f}  F1 Discharge: {f1_both[0]:.4f}"  # type: ignore
            f"\nF1 Macro: {f1_macro:.4f}\n"
        )

        if eval_loss < best_eval_loss and tol < abs(eval_loss - best_eval_loss):
            logger.info(f"Epoch {epoch + 1} best val loss so far: {eval_loss:.4f}")
            best_eval_loss = eval_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = {k: v.clone() for k, v in seq_model.state_dict().items()}
        elif eval_loss > best_eval_loss or tol > abs(eval_loss - best_eval_loss):
            epochs_no_improve += 1
            logger.info(
                f"Epoch {epoch + 1} no improvement. {patience - epochs_no_improve} epochs left before stopping"
            )

        if epochs_no_improve == patience:
            logger.info(
                f"No improvement for {patience} epochs. Stopping and reverting to epoch {best_epoch}"
            )
            break

    if best_model_state is not None:
        seq_model.load_state_dict(best_model_state)
        logger.info(f"Restored model from epoch {best_epoch} (val loss: {best_eval_loss:.4f})")

    # -- Combine val + test for step-by-step inference ------------------------
    # pad_stays uses groupby('ed_stay_id') which sorts ascending -- mirror that
    # here so combined_dataset row order matches combined_tensor_dataset order
    print("Combining val and test datasets to be used for step by step predictions")
    combined_dataset = pd.concat([
        df_val.sort_values('ed_stay_id'),
        df_test.sort_values('ed_stay_id'),
    ]).reset_index(drop=True)

    combined_tensor_dataset = ConcatDataset([pad_val.dataset, pad_test.dataset])
    combined_loader = DataLoader(combined_tensor_dataset, batch_size=batch_size, shuffle=False)

    n_stays_df = combined_dataset['ed_stay_id'].nunique()
    n_stays_tensors = len(combined_tensor_dataset)
    if n_stays_df != n_stays_tensors:
        msg = f"Stay count mismatch: DataFrame has {n_stays_df}, tensors have {n_stays_tensors}"
        logger.error(msg)
        raise AssertionError(msg)

    # -- Step-by-step inference -----------------------------------------------
    step_model = StepByStepWrapper(seq_model).to(device)
    probs_list = []
    lengths_list = []
    print("Running model in evaluation mode to generate predictions, stand by...")
    step_model.eval()
    with torch.no_grad():
        for X, y, lengths in combined_loader:
            X = X.to(device)
            logits = step_model(X, lengths)
            probs = torch.softmax(logits, dim=-1).cpu()  # (B, T_max, 2)
            pad_needed = pad_length - probs.shape[1]
            if pad_needed > 0:
                probs = torch.nn.functional.pad(probs, (0, 0, 0, pad_needed))
            probs_list.append(probs)
            lengths_list.append(lengths)

    final_probs = torch.cat(probs_list, dim=0)   # (N, pad_length, 2)
    full_lengths = torch.cat(lengths_list, dim=0) # (N,)

    icu_probs = final_probs[:, :, 1]
    t = torch.arange(pad_length).unsqueeze(0)
    mask = t < full_lengths.unsqueeze(1)
    combined_dataset['p_icu'] = icu_probs[mask].numpy()

    # -- Save -----------------------------------------------------------------
    artifact_dir = PROJECT_ROOT / artifact_cfg["sbs_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_path = artifact_dir / f"sbs_predictions_{timestamp}.parquet"
    combined_dataset.to_parquet(predictions_path, index=False)
    logger.info(f"Saved sbs predictions -> {predictions_path.name}")

    sbs_hf = hf_cfg['sbs_data']
    from datasets import Dataset
    ds = Dataset.from_pandas(combined_dataset, preserve_index=False)
    ds.push_to_hub(
        hf_cfg['step_by_step'],
        config_name=sbs_hf['config_name'],
        split=sbs_hf['split_name'],
        data_dir=sbs_hf['data_dir'],
    )
    logger.info(f"Pushed sbs predictions to {hf_cfg['step_by_step']} / {sbs_hf['config_name']}")