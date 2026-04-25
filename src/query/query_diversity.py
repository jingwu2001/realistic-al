from typing import Tuple, Optional, Union
from tqdm import tqdm
import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import stats
from torch.utils.data import DataLoader

from .kcenterGreedy import KCenterGreedy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bayesian_module import BayesianModule


NAMES = ["kcentergreedy", "badge", "vendi"]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def query_sampler(
    cfg: DictConfig,
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    unlabeled_dataloader: DataLoader,
    acq_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Computes queries for all diversity based query methods.
    IMPLEMENTATION: Add diversity queries here.

    Args:
        cfg (DictConfig): config
        model (torch.nn.Module): model for queries.
        labeled_dataloader (DataLoader): carries labeled data.
        unlabeled_dataloader (DataLoader): carries unlabeled data.
        acq_size (int): size of query in datapoints.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: [pool_indices, ranking values, extra_info]
    """
    name = cfg.query.name
    ### IMPLEMENTATION
    # Add novel query methods here
    if name == "kcentergreedy":
        indices = _get_kcg(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        # there is no ranking, therefore we add descending numerics as ranking values
        return indices, np.arange(acq_size)[::-1]
    elif name == "badge":
        indices = _get_badge(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        # there is no ranking, therefore we add descending numerics as ranking values
        return indices, np.arange(acq_size)[::-1]
    elif name == "vendi":
        indices, scores, extra_info = _get_vendi(
            cfg, model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )

        return indices, scores, extra_info
    else:
        raise NotImplementedError


def init_centers(X: np.ndarray, K: int) -> np.ndarray:
    """Determines k centers using KMeans++ of data.
    Source: https://github.com/decile-team/distil/blob/main/distil/active_learning_strategies/badge.py

    Args:
        X (np.ndarray): NxD data
        K (int): #centers to be determined

    Returns:
        np.ndarray: Indices of centers
    """
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
        else:
            newD = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def chunked_pdist(
    x: torch.Tensor, y: torch.Tensor, device: str = "cuda:0", max_size: int = 8
) -> torch.Tensor:
    """Compute Pairwise Distance for chunks given a maximal size of each chunk part.

    Args:
        x (torch.Tensor): input tensor NxD
        y (torch.Tensor): input tensor MxD
        device (str, optional): device for computation. Defaults to "cuda:0".
        max_size (int, optional): maximum size of chunk of x for pdist computation in GB. Defaults to 8.

    Returns:
        torch.Tensor: NxM pairwise distance matrix
    """
    pdist = torch.nn.PairwiseDistance(p=2)
    new_shape = x.shape[0]
    while (x.element_size() * x.nelement() / 10e9) * (
        new_shape / x.shape[0]
    ) > max_size:
        new_shape /= 2

    y_dev = y.to(device)
    ind_start = 0
    new_shape = x.shape[0] / 2
    dists = []
    while ind_start < x.shape[0]:
        ind_offset = int(min(new_shape, x.shape[0] - ind_start))
        dist_part = pdist(x[ind_start : ind_start + ind_offset].to(device), y_dev).to(
            "cpu"
        )
        dists.append(dist_part)
        ind_start += ind_offset
    dists = torch.concat(dists)
    assert dists.shape[0] == x.shape[0]
    return dists


def get_grad_embedding(
    model: torch.nn.Module,
    dataloader: DataLoader,
    grad_embedding_type: str = "linear",
    device: str = "cuda:0",
) -> torch.Tensor:
    """Compute gradient embedding of final layer of model for each input in dataloader.

    Args:
        model (torch.nn.Module): requires get_features and classifer
        dataloader (DataLoader): contains data.
        grad_embedding_type (str, optional): in [bias, linear,...]. Defaults to "linear".
        device (str, optional): device for gradient computation. Defaults to "cuda:0".

    Returns:
        torch.Tensor: Embedding of last layer gradient
    """

    BayesianModule.k = 1
    start_index = 0
    assert hasattr(model, "model")  # model requires sub class model (due to code)
    assert hasattr(
        model.model, "classifier"
    )  # model.model requires classification head (due to code).
    for i, (x, y) in enumerate(dataloader):
        if isinstance(x, (list, tuple)):
            inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in x]
        else:
            inputs = x.to(device)
        with torch.no_grad():
            features = model.get_features(inputs)  # B x Z
        l1 = features  # B x Z
        if not model.hparams.model.small_head and isinstance(model.model.classifier, nn.Sequential):
            l1 = model.model.classifier[:-1](features)
        embDim = l1.shape[-1]
        outputs = model.model.classifier(features)

        preds = torch.argmax(outputs, dim=1)

        loss = F.cross_entropy(outputs, preds, reduction="sum")
        l0_grads = torch.autograd.grad(loss, outputs)[0]  # B x C

        # Calculate the linear layer gradients as well if needed
        if grad_embedding_type != "bias":
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, outputs.shape[-1])  # B x C*Z

        # Populate embedding tensor according to the supplied argument.
        if grad_embedding_type == "bias":
            gradient_embedding = l0_grads
        elif grad_embedding_type == "linear":
            gradient_embedding = l1_grads
        else:
            gradient_embedding = torch.cat([l0_grads, l1_grads], dim=1)
        if i == 0:
            gradient_embeddings = torch.empty(
                [len(dataloader.dataset), gradient_embedding.shape[1]], device="cpu"
            )
        gradient_embeddings[
            start_index : start_index + gradient_embedding.shape[0]
        ] = gradient_embedding.to("cpu")
        start_index += gradient_embedding.shape[0]
        torch.cuda.empty_cache()
    return gradient_embeddings


def _get_badge(
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 100,
):
    assert hasattr(model, "get_features")  # model requires function get_features
    grad_embedding = get_grad_embedding(model, pool_loader)
    grad_embedding = grad_embedding.numpy()
    acq_indices = init_centers(grad_embedding, acq_size)
    return np.array(acq_indices)


def _get_kcg(
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 100,
):
    """Returns the indices of the core-set for the pool of the model via the k-center Greedy approach."""
    assert hasattr(model, "get_features")  # model requires function get_features
    with torch.no_grad():
        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in labeled_dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in inputs]
            else:
                inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_labeled = features.detach().cpu().numpy()

        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in pool_loader:
            if isinstance(inputs, (list, tuple)):
                inputs = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in inputs]
            else:
                inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_unlabeled = features.detach().cpu().numpy()

        feat_merge = np.concatenate([feat_labeled, feat_unlabeled], axis=0)
        indices_labeled = np.arange(feat_labeled.shape[0])
        del features
        del feat_labeled
        del feat_unlabeled
        sampling = KCenterGreedy(feat_merge)
        acq_indices = np.array(sampling.select_batch_(indices_labeled, acq_size))

        # subtract the indices of the labeled data to get pool indices
        acq_indices -= indices_labeled.shape[0]
    return acq_indices

@torch.no_grad()
def _get_vendi(
    cfg,
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 128,
):
    assert hasattr(model, "get_features")  # model requires function get_features

    normalization = cfg.query.vendi.normalization

    feat_labeled = get_embeddings(model, labeled_dataloader, cpu=False, numpy=False)
    feat_unlabeled = get_embeddings(model,pool_loader, cpu=False, numpy=False)

    feat_labeled, feat_unlabeled = normalize_features(feat_labeled, feat_unlabeled, normalization)
    # feat_labeled: (num_labeled, feature_dim)
    # feat_unlabeled: (num_unlabeled, feature_dim)

    scores = vendi_from_features(cfg, feat_labeled, feat_unlabeled)
    # scores: (num_unlabeled,)
    
    # Sort scores descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    acq_indices = sorted_indices[:acq_size]
    
    extra_info_dict = calculate_extra_info(acq_size, sorted_scores)

    return acq_indices, sorted_scores[:acq_size], extra_info_dict



def normalize_features(
    feat_labeled: torch.tensor, 
    feat_unlabeled: torch.tensor, 
    normalization: str
    ) -> torch.tensor:

    normalization = normalization.lower()

    if normalization == "none":
        return feat_labeled, feat_unlabeled

    if normalization == "l2":
        feat_labeled = F.normalize(feat_labeled, p=2, dim=1)
        feat_unlabeled = F.normalize(feat_unlabeled, p=2, dim=1)

    elif normalization == 'minmax':
        feat_all = torch.cat([feat_labeled, feat_unlabeled], dim=0)
        min_all = torch.min(feat_all, dim=0)[0]
        max_all = torch.max(feat_all, dim=0)[0]
        diff = max_all - min_all
        # Protect against division by zero
        diff[diff == 0] = 1.0
        feat_labeled = (feat_labeled - min_all) / diff
        feat_unlabeled = (feat_unlabeled - min_all) / diff

    elif normalization == 'zscore':
        feat_all = torch.cat([feat_labeled, feat_unlabeled], dim=0)
        mean_all = torch.mean(feat_all, dim=0)
        std_all  = torch.std(feat_all , dim=0)
        # Protect against division by zero
        std_all[std_all == 0] = 1.0
        feat_labeled = (feat_labeled - mean_all) / std_all
        feat_unlabeled = (feat_unlabeled - mean_all) / std_all

    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    return feat_labeled, feat_unlabeled

def rbf_kernel(x, y=None, gamma=None):
    """
    x: (N, D)
    y: (M, D)
    gamma: float
    """
    if y is None:
        y = x
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    x_norm = (x**2).sum(dim=1).view(-1, 1)
    y_norm = (y**2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * (x @ y.T)
    return torch.exp(-gamma * dist)

def cosine_similarity(x, y=None):
    """
    x: (N, D)
    y: (M, D)
    """
    if y is None:
        y = x
    return (x @ y.T) / (torch.norm(x, dim=1, p=2).view(-1, 1) * torch.norm(y, dim=1, p=2).view(1, -1))    



def renyi_entropy(p: torch.Tensor, q: float) -> torch.Tensor:
    """
    Assume the input is (N, D),
    where N is the batch size and D is the dimension of the distribution
    """
    assert q > 0, "q must be > 0"

    if math.isclose(q, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return torch.special.entr(p).sum(dim=1)

    return p.pow(q).sum(dim=1).log() / (1 - q)

def calculate_extra_info(acq_size, sorted_scores):
    """
    store the maximum and minimmum scores of acquired samples, and also the max, min, and medium of unacquried samples

    Args
    sorted_scores: SORTED acq scores
    """

    if isinstance(sorted_scores, torch.Tensor):
        sorted_scores = sorted_scores.detach().cpu().numpy()
    acquired = sorted_scores[:acq_size]
    unacquried = sorted_scores[acq_size:]  # fixed off-by-one (was acq_size + 1)

    funcs = [np.max, np.min, np.median]
    acquired_stats = (
        np.array([func(acquired) for func in funcs])
        if len(acquired) > 0
        else np.full(len(funcs), np.nan)
    )
    unacquried_stats = (
        np.array([func(unacquried) for func in funcs])
        if len(unacquried) > 0
        else np.full(len(funcs), np.nan)
    )

    results = {
        "acq": acquired_stats,
        "else": unacquried_stats
    }

    return results

@torch.no_grad()
def get_embeddings(model, loader, cpu=False, numpy=False):
    features = torch.tensor([]).to(DEVICE)
    for inputs, _ in loader:
        if isinstance(inputs, (list, tuple)):
            inputs = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in inputs]
        else:
            inputs = inputs.to(DEVICE)
        features_batch = model.get_features(inputs)
        features = torch.cat((features, features_batch), 0)
    features = features.detach()
    if numpy:
        features = features.cpu().numpy()
    elif cpu:
        features = features.cpu()
    return features

def vendi_from_features(cfg, feat_labeled, feat_unlabeled):
    """
    Args:
        cfg: config
        feat_labeled: (num_labeled, feature_dim)
        feat_unlabeled: (num_unlabeled, feature_dim)
    Returns:
        scores: (num_unlabeled,)

    Intermediates:
        K_LL: (L, L)
        K_UL: (U, L)
        K: (batch_size, L + 1, L + 1)
        
        In each iteration the algorithm takes batch_size samples from K_UL and paste it into off-diagonal blocks,
        and diagonalize the matrices
    """
    vendi_cfg = cfg.query.vendi
    gamma = vendi_cfg.gamma
    q = vendi_cfg.q
    batch_size = vendi_cfg.batch_size if vendi_cfg.batch_size is not None else 200

    L = feat_labeled.shape[0]
    U = feat_unlabeled.shape[0]

    # K_LL and K_UL are kernel matrices

    K_LL = compute_kernel_matrix(cfg, feat_labeled, feat_labeled)
    K_LL = torch.as_tensor(K_LL, dtype=torch.float64, device=DEVICE)
    K_UL = compute_kernel_matrix(cfg, feat_unlabeled, feat_labeled)
    K_UL = torch.as_tensor(K_UL, dtype=torch.float64, device=DEVICE)

    K = torch.empty(batch_size, L + 1, L + 1, device=DEVICE, dtype=torch.float64)
    K[:, :L, :L] = K_LL
    K[:, L, L] = 1.0
    scores = torch.empty(U, device=DEVICE, dtype=torch.float64)

    n_eig = int((L + 1) / 3) if vendi_cfg.approx else L + 1
    eig_vals = torch.zeros(U, n_eig, device=DEVICE, dtype=torch.float64)

    num_iter = math.ceil(U / batch_size)
    for i in tqdm(range(num_iter), desc="Calculating Vendi scores"):
        K_UL_batch = K_UL[batch_size * i: batch_size * (i + 1)]
        current_batch_size = K_UL_batch.shape[0]
        K_batch = K[:current_batch_size]

        K_batch[:, L, :L] = K_UL_batch
        K_batch[:, :L, L] = K_UL_batch

        if vendi_cfg.approx:
            k = int(K_batch.shape[1] / 3)
            ev, _ = torch.lobpcg(K_batch, k=k)
            ev = ev.clamp(min=0)
            ev_sum = ev.sum(dim=1, keepdim=True)
            ev = ev / ev_sum.clamp(min=1e-12)
        else:
            ev = torch.linalg.eigvalsh(K_batch).clamp(min=0) / (L + 1)
        eig_vals[batch_size * i: batch_size * (i + 1)] = ev
        entropy = renyi_entropy(ev, q)

        scores[batch_size * i: batch_size * (i + 1)] = entropy

    max_diff = (eig_vals.max(dim=0).values - eig_vals.min(dim=0).values).max().item()
    if max_diff < 1e-8:
        print("[WARNING] all samples have identical eigenvalue distributions — approximation may be degenerate.")
    else:
        print(f"[INFO] eigenvalue distributions: largest difference across samples = {max_diff:.6e}")

    del K, K_LL, K_UL
    torch.cuda.empty_cache()
    
    scores = scores.exp().detach().cpu().numpy()
    return scores

def compute_kernel_matrix(
    cfg: DictConfig, 
    x: Union[np.ndarray, torch.tensor], 
    y: Union[np.ndarray, torch.tensor],
) -> torch.Tensor:
    """
    Compute the kernel matrix in batch style.
    DOES NOT PERFORM NORMALIZATION

    feat_labeled: (L, D)
    feat_unlabeled: (U, D)
    """
    vendi_cfg = cfg.query.vendi

    if vendi_cfg.kernel == 'rbf':
        return rbf_kernel(x, y, gamma=vendi_cfg.gamma)
    elif vendi_cfg.kernel == 'cosine':
        return cosine_similarity(x, y)
    else:
        raise ValueError(f"Unknown kernel: {vendi_cfg.kernel}")


# TODO: 
# 0. add docstring indicating input/output/variable shapes (Done)
# 1. implement cosine kernel logic  (Done)
# 2. refactor the code
# 3. sweep different kernel and hyperparameters (rbf gamma)
# 4. add more options to vendi config
# 5. debug 

# 6. implement time series

if __name__ == '__main__':
    x = torch.rand(3, 10)
    y = torch.rand(2, 10)
    cfg = DictConfig({
        "query": {
            "vendi": {
                "gamma": 1.0,
                "q": 1.0,
                "batch_size": 200,
                "kernel": "rbf"
            }
        }
    })
    scores = vendi_from_features(cfg, x, y)
    print(scores)
