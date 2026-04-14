"""
modeling/lstm/training_eval_loops.py

Training and evaluation loop functions for the LSTM sequence classifier.
Imported by train.py and any other scripts (e.g. hyperparameter tuning)
that need to run the same loop logic.
"""

import torch


def training_loop(train_data_loader, model, optimizer, loss_fn, device, training_update):
    """
    Run one full training epoch over train_data_loader.

    Args:
        train_data_loader: DataLoader yielding (X, y, lengths) batches.
        model:             LSTM model in training mode.
        optimizer:         Configured optimizer instance.
        loss_fn:           Loss function (e.g. CrossEntropyLoss with class weights).
        device:            "cuda", "mps", or "cpu".
        training_update:   Print a progress line every N batches.

    Returns:
        Average training loss over all batches.
    """
    num_batches = len(train_data_loader)
    total_loss = 0

    model.train()
    for batch, (X, y, lengths) in enumerate(train_data_loader):
        X, y = X.to(device), y.to(device)  # leave lengths on CPU for pack_padded_sequence
        optimizer.zero_grad()
        preds = model(X, lengths)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % training_update == 0:
            print(f"Current loss {loss.item():>5.4f}  [Batch {batch:,} / {num_batches:,}]")

    return total_loss / num_batches


def evaluation_loop(test_data_loader, model, loss_fn, device):
    """
    Run inference over test_data_loader with gradients disabled.

    Args:
        test_data_loader: DataLoader yielding (X, y, lengths) batches.
        model:            LSTM model.
        loss_fn:          Loss function (same as training, for comparable loss values).
        device:           "cuda", "mps", or "cpu".

    Returns:
        Tuple of (avg_loss, predictions, true_labels) where predictions is a
        (N, num_classes) softmax tensor and true_labels is a (N,) integer tensor.
    """
    num_batches = len(test_data_loader)
    test_loss = 0
    all_preds = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for X, y, lengths in test_data_loader:
            X, y = X.to(device), y.to(device)  # leave lengths on CPU
            preds = model(X, lengths)  # (batch_size, num_classes) logits
            loss = loss_fn(preds, y)
            test_loss += loss.item()
            all_preds.append(torch.softmax(preds, dim=-1).cpu())
            true_labels.append(y.cpu())

    test_loss /= num_batches
    final_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(true_labels, dim=0)
    return test_loss, final_preds, all_labels
