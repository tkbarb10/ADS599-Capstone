"""
modeling/lstm/train.py

Trains an LSTM sequence classifier using hyperparameters from
modeling/config/lstm.yaml.  Full pipeline:

  1. Load full_patient_state from HuggingFace
  2. Remove outlier stays
  3. Stratified 80/10/10 train/test/val split by subject_id
  4. StandardScaler fit on train, transform all splits
  5. Zero-pad sequences and build DataLoaders
  6. Train with early stopping monitored on val loss
  7. Restore best model, evaluate on held-out test set
  8. Save weights, scaler, test predictions, and epoch losses

Usage:
    python -m modeling.lstm.train
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from modeling.data_prep.lstm import (
    load_and_prep_lstm,
    remove_outlier_stays,
    split_data,
    scaling,
    pad_data,
)
from modeling.lstm.model import LSTMSequenceModel
from modeling.lstm.training_eval_loops import training_loop, evaluation_loop
from utils.load_yaml_helper import load_yaml
from utils.logging_helper import setup_logging
from typing import Optional

settings = load_yaml("project_setup/settings.yaml")
cfg = load_yaml("modeling/config/lstm.yaml")
logger = setup_logging(settings["logging"]["train_path"])

PROJECT_ROOT = Path(__file__).parents[2]

# -- Config -------------------------------------------------------------------
random_state = cfg["random_state"]

model_cfg = cfg["model"]

train_cfg = cfg["training"]
epochs = train_cfg["epochs"]
learning_rate = train_cfg["learning_rate"]
weight_decay = train_cfg["weight_decay"]
optimizer_key = train_cfg["optimizer"]
training_update = train_cfg["training_update"]
patience = train_cfg["patience"]
tol = train_cfg["tol"]

artifact_cfg = cfg["artifacts"]
hf_cfg = settings["hugging_face"]

optimizer_options = {
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
}

# -- Reproducibility ----------------------------------------------------------
# torch.cuda.manual_seed_all omitted -- only needed for multi-GPU
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
# Pipeline functions
# =============================================================================

def prep_data(hf_cfg: dict, device: str, sample_size: Optional[float]=None) -> tuple:
    """
    Load, filter, split, compute class weights, scale, and pad data.

    Args:
        hf_cfg: hugging_face section of settings.yaml.
        device: torch device string for placing class_weight tensor.
        sample_size: optional argument to add to only train on a fraction of the data.  Useful if working with limited RAM

    Returns:
        (pad_train, pad_val, pad_test, scaler, df_test, class_weight)
        pad_*  -- DataLoaders ready for the model
        scaler -- fitted StandardScaler (saved as artifact)
        df_test -- raw test DataFrame for building predictions file
        class_weight -- weighted tensor for CrossEntropyLoss
    """
    import gc

    df, state_cols = load_and_prep_lstm(hf_cfg=hf_cfg)

    if sample_size is not None and not (0 < sample_size < 1):
        raise ValueError(f"sample_size must be a float between 0 and 1, got {sample_size}")
    if sample_size is not None:
        df = df.sample(frac=sample_size, random_state=random_state)

    modeling_df = remove_outlier_stays(df=df)
    del df
    gc.collect()

    df_train, df_test, df_val = split_data(modeling_df)
    del modeling_df
    gc.collect()

    stay_labels_train = df_train.drop_duplicates("ed_stay_id")["label"].values
    labels_all = torch.tensor(stay_labels_train)
    n_discharge = (labels_all == 0).sum().float()
    n_icu = (labels_all == 1).sum().float()
    class_weight = torch.tensor([1.0, (n_discharge / n_icu).item()]).to(device)
    logger.info(f"Class weights: discharge={class_weight[0]:.2f}  icu={class_weight[1]:.2f}")

    # scaling modifies in place -- scaled_* and df_* are the same objects
    scaled_train, scaled_test, scaled_val, scaler = scaling(
        train=df_train, test=df_test, val=df_val
    )
    del df_train, df_val  # df_test kept for save_artifacts; scaled_test IS df_test
    gc.collect()

    logger.info("Padding data, this might take a few min...")
    pad_train, pad_test, pad_val = pad_data(
        train=scaled_train, test=scaled_test, val=scaled_val, state_cols=state_cols
    )
    del scaled_train, scaled_val  # scaled_test kept (same object as df_test)
    gc.collect()

    return pad_train, pad_val, pad_test, scaler, df_test, class_weight


def train_model(
    pad_train, pad_val, pad_test, class_weight: torch.Tensor, device: str
) -> tuple:
    """
    Build model, train with early stopping on val loss, restore best weights,
    and run final evaluation on the held-out test set.

    Args:
        pad_train: Training DataLoader.
        pad_val: Validation DataLoader (used for early stopping).
        pad_test: Test DataLoader (evaluated once after training).
        class_weight: Tensor of per-class weights for CrossEntropyLoss.
        device: torch device string.

    Returns:
        (seq_model, train_losses, eval_losses, test_preds, test_true)
        seq_model  -- trained model with best weights loaded
        train_losses -- list of average training loss per epoch
        eval_losses -- list of average val loss per epoch
        test_preds -- (N, 2) softmax tensor from held-out test set
        test_true -- (N,) integer label tensor from held-out test set
    """
    seq_model = torch.compile(LSTMSequenceModel().to(device))
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

    _, test_preds, _ = evaluation_loop(
        test_data_loader=pad_test,
        model=seq_model,
        loss_fn=loss_fn,
        device=device,
    )

    return seq_model, train_losses, eval_losses, test_preds


def save_artifacts(
    seq_model: nn.Module,
    scaler,
    test_preds: torch.Tensor,
    df_test: pd.DataFrame,
    train_losses: list,
    eval_losses: list,
) -> None:
    """
    Save model weights, scaler, test predictions, and epoch losses to artifacts/lstm/.

    Test predictions are joined to stay-level metadata from df_test so the
    output parquet matches the format expected by evaluate.py.

    Epoch losses are saved as a parquet with columns [epoch, train_loss, eval_loss]
    for plotting learning curves.
    """
    artifact_dir = PROJECT_ROOT / artifact_cfg["dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weights_path = artifact_dir / f"lstm_weights_{timestamp}.pt"
    scaler_path = artifact_dir / f"lstm_scaler_{timestamp}.pkl"
    predictions_path = artifact_dir / f"test_predictions_{timestamp}.parquet"
    losses_path = artifact_dir / f"training_losses_{timestamp}.parquet"

    torch.save(seq_model.state_dict(), weights_path)
    logger.info(f"Saved model weights -> {weights_path.name}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler -> {scaler_path.name}")

    # Build predictions DataFrame -- stay order matches groupby("ed_stay_id") sort
    test_meta = (
        df_test[["ed_stay_id", "subject_id", "label"]]
        .drop_duplicates("ed_stay_id")
        .sort_values("ed_stay_id")
        .reset_index(drop=True)
    )
    test_meta["pred_prob"] = test_preds[:, 1].numpy()  # P(ICU)
    test_meta["pred_label"] = test_preds.argmax(dim=1).numpy()
    test_meta.to_parquet(predictions_path, index=False)
    logger.info(f"Saved test predictions -> {predictions_path.name}")

    losses_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "eval_loss": eval_losses,
    })
    losses_df.to_parquet(losses_path, index=False)
    logger.info(f"Saved epoch losses -> {losses_path.name}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training loop to train a LSTM classifier")
    parser.add_argument(name="--sample_size", required=False, type=float)
    args = parser.parse_args()

    logger.info(f"Using device: {device}")
    pad_train, pad_val, pad_test, scaler, df_test, class_weight = prep_data(hf_cfg, device, sample_size=args.sample_size)
    seq_model, train_losses, eval_losses, test_preds = train_model(
        pad_train, pad_val, pad_test, class_weight, device
    )
    save_artifacts(seq_model, scaler, test_preds, df_test, train_losses, eval_losses)
