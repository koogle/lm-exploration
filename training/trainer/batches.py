"""Padding token rows into batches for training and scoring."""

from training_data.subword import LOSS_IGNORE_INDEX


def collate(rows, device):
    """Inputs and targets of rows `{"x": ids, "y": targets}`, right-padded to the longest:
    inputs with 0, targets with `LOSS_IGNORE_INDEX`."""
    import torch

    width = max(len(row["x"]) for row in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full_like(x, LOSS_IGNORE_INDEX)
    for i, row in enumerate(rows):
        x[i, : len(row["x"])] = torch.tensor(row["x"], device=device)
        y[i, : len(row["y"])] = torch.tensor(row["y"], device=device)
    return x, y
