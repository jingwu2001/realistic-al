
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

if __name__ == "__main__":
    test_vendi()
