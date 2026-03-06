import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np

DEVICE = torch.device('cuda')

@torch.no_grad()
def get_confusion_matrix(model: nn.Module, dataloader: DataLoader, k: int = None, normalize_type: str = None) -> np.ndarray:
    model.eval()
    pred = []
    gt = []
    model.to(DEVICE)
    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = model(x, k=k)
        pred.append(y_hat.argmax(dim=1))
        gt.append(y)
    pred = torch.cat(pred).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    return confusion_matrix(gt, pred, normalize=normalize_type)