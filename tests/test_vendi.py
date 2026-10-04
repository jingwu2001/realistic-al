
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("src"))

from query.query_diversity import _get_vendi, DEVICE

# Mock Model
class MockModel(torch.nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.feature_dim = feature_dim

    def get_features(self, x):
        # Return random features
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.feature_dim).to(x.device)

def test_vendi():
    print("Testing _get_vendi...")
    
    # Configuration
    cfg = OmegaConf.create({
        "query": {
            "name": "vendi",
            "vendigamma": 1.0,
            "vendi": {
                "batch_size": 32,
                "normalization": "l2",
                "q": 1.0
            }
        }
    })
    
    # Use the device defined in query_diversity.py or fallback
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Mock Data
    feature_dim = 16
    labeled_data = torch.randn(50, feature_dim)
    labeled_targets = torch.randint(0, 10, (50,))
    labeled_dataset = TensorDataset(labeled_data, labeled_targets)
    labeled_loader = DataLoader(labeled_dataset, batch_size=10)
    
    pool_data = torch.randn(200, feature_dim)
    pool_targets = torch.randint(0, 10, (200,))
    pool_dataset = TensorDataset(pool_data, pool_targets)
    pool_loader = DataLoader(pool_dataset, batch_size=10)
    
    model = MockModel(feature_dim).to(device)
    
    acq_size = 20
    
    # Run _get_vendi
    try:
        indices, scores = _get_vendi(cfg, model, labeled_loader, pool_loader, acq_size=acq_size)
        
        print(f"Indices shape: {indices.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Indices: {indices}")
        
        assert len(indices) == acq_size, f"Expected {acq_size} indices, got {len(indices)}"
        assert len(scores) == acq_size, f"Expected {acq_size} scores, got {len(scores)}"
        
        print("Test PASSED!")
        
    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

def test_boundary_gap_and_ties():
    import numpy as np
    from query.query_diversity import _boundary_tie_stats

    # clean cutoff: batch {0, 1} beats the best left-out score (2.0) by 1.0
    s = np.array([5.0, 3.0, 2.0, 1.0])
    st = _boundary_tie_stats(s, np.array([0, 1]), 2)
    assert st["gap"] == 1.0 and st["rel_gap"] == 1.0 / 3.0
    assert st["n_tied"] == 1 and st["n_tied_excluded"] == 0

    # cutoff inside a tie group of 3: one tied sample is picked by tie-break
    # alone and two are excluded by luck
    s = np.array([5.0, 3.0, 3.0, 3.0, 1.0])
    st = _boundary_tie_stats(s, np.array([0, 2]), 2)
    assert st["gap"] == 0.0 and st["rel_gap"] == 0.0
    assert st["n_tied"] == 3 and st["n_tied_acquired"] == 1 and st["n_tied_excluded"] == 2

    # whole pool acquired -> no boundary
    st = _boundary_tie_stats(s, np.arange(5), 5)
    assert st["gap"] is None and st["n_tied_excluded"] == 0


def test_boundary_metrics_logged():
    import numpy as np
    from query.query_diversity import _boundary_tie_stats
    from utils.wandb_utils import _query_score_metrics

    st = _boundary_tie_stats(np.array([5.0, 3.0, 3.0, 1.0]), np.array([0, 1]), 2)
    m = _query_score_metrics({"boundary_ties": st})
    assert m["al/query/boundary_gap"] == 0.0
    assert m["al/query/boundary_value"] == 3.0
    assert m["al/query/boundary_n_tied_excluded"] == 1.0

    # no-boundary case logs nothing for the None fields
    st = _boundary_tie_stats(np.array([5.0, 3.0]), np.arange(2), 2)
    m = _query_score_metrics({"boundary_ties": st})
    assert "al/query/boundary_gap" not in m
    assert m["al/query/boundary_n_tied"] == 0.0


if __name__ == "__main__":
    test_vendi()
