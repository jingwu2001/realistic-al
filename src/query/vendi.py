from .query_diversity import *

def _get_vendi_q2_sequential(
    cfg,
    model: torch.nn.Module,
    labeled_dataloader: DataLoader,
    pool_loader: DataLoader,
    acq_size: int = 128,
    ):
    """Vendi score special case with q=2.
    Eliminates the need of calculating eigenvalues (see vendi_q2_from_features).

    Same acquisition contract as _get_vendi: returns
        (acq_indices, sorted_scores[:acq_size], extra_info_dict).
    """
    vendi_cfg = cfg.query.vendi
    

    emb_labeled, emb_unlabeled, grad_norms = get_vendi_embeddings(
        vendi_cfg, model, labeled_dataloader, pool_loader
    )

    emb_labeled, emb_unlabeled = normalize_features(
        emb_labeled, emb_unlabeled, vendi_cfg.normalization
    )
    # emb_labeled: (num_labeled, emb_dim)
    # emb_unlabeled: (num_unlabeled, emb_dim)

    indices = vendi_q2_sequential_from_features(
        vendi_cfg, emb_labeled, emb_unlabeled, acq_size
    )
    # rank_scores / true_scores: (num_unlabeled,)

    del emb_labeled, emb_unlabeled
    torch.cuda.empty_cache()

    return indices


def vendi_q2_sequential_from_features(vendi_cfg, feat_labeled, feat_unlabeled, acq_size):
    """Calculates the Vendi score with Renyi q=2 from the labeled and unlabeled features
    """
    if vendi_cfg.kernel == 'linear':
        raise ValueError("Unsupported kernel: linear")
    # gamma only matters for rbf; skip the pairwise-distance heuristic otherwise
    gamma = (
        resolve_gamma(vendi_cfg.gamma, feat_labeled)
        if vendi_cfg.kernel == "rbf"
        else None
    )
    batch_size = vendi_cfg.batch_size

    feat_labeled = feat_labeled.to(DEVICE, torch.float64)      # (L, d)
    feat_unlabeled = feat_unlabeled.to(DEVICE, torch.float64)  # (U, d)

    # The score is (\sum p_i^2)^-1
    # proportional to (\sum K_ij^2)^-1
    # so we only need to pick the smallest \sum K_ij^2
    K_UL = compute_kernel_matrix(vendi_cfg.kernel, gamma, feat_unlabeled, feat_labeled) # (U, L)
    K_UL = torch.as_tensor(K_UL, dtype=torch.float64, device=DEVICE)
    assert torch.isfinite(K_UL).all(), "non-finite entries in K_UL"

    # sum of non-negative squares -> no cancellation; exclude the constant
    # ||K_LL||_F^2 so the varying signal is not swamped in finite precision.
    rank_scores = -(K_UL ** 2).sum(dim=1) # (U, ), inversely proportional to the actual scores

    chosen_indices = []
    while len(chosen_indices) < acq_size:
        bs = min(batch_size, acq_size - len(chosen_indices))
        _, idx = torch.topk(rank_scores, bs)
        chosen_indices.extend(idx.tolist())

        # Treat the freshly chosen samples as if they were just added to the
        # labeled set: subtract their k(chosen, unlabeled)^2 contribution.
        feat_chosen = feat_unlabeled[idx]  # (bs, d); advanced indexing already copies
        K_batch = compute_kernel_matrix(
            vendi_cfg.kernel, gamma, feat_unlabeled, feat_chosen
        )  # (U, bs)
        rank_scores -= (K_batch ** 2).sum(dim=1)  # (U,)

        # Mask the chosen rows so the next topk can never re-select them.
        rank_scores[idx] = float("-inf")

    del K_UL
    torch.cuda.empty_cache()

    return chosen_indices
