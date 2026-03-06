
import torch
import numpy as np
import sklearn.metrics
from src.query import query_diversity
from src.query.query_bandit import calculate_vendi_score

# Mock query_diversity.renyi_entropy since we can't easily import it without setup if we run standalone, 
# but here we rely on the project environment. 
# ensure query_diversity is imported correctly in query_bandit. 
# It seems query_bandit imports query_diversity.

def test_calculate_vendi_score():
    torch.manual_seed(42)
    feats = torch.randn(10, 5)
    feats_np = feats.numpy()
    
    # Test RBF
    print("Testing RBF...")
    score_rbf = calculate_vendi_score(feats, gamma=1.0, q=1.0, kernel='rbf')
    print(f"RBF Score: {score_rbf}")
    
    # Compare with manual
    K_rbf = sklearn.metrics.pairwise.rbf_kernel(feats_np, gamma=1.0)
    K_rbf /= K_rbf.shape[0]
    ev_rbf = torch.linalg.eigvalsh(torch.tensor(K_rbf))
    entropy_rbf = query_diversity.renyi_entropy(ev_rbf.unsqueeze(0), 1.0).item()
    expected_rbf = np.exp(entropy_rbf)
    print(f"Expected RBF: {expected_rbf}")
    assert np.isclose(score_rbf, expected_rbf), "RBF Mismatch"

    # Test Cosine
    print("\nTesting Cosine...")
    score_cos = calculate_vendi_score(feats, gamma=1.0, q=1.0, kernel='cosine')
    print(f"Cosine Score: {score_cos}")
    
    # Compare with manual
    K_cos = sklearn.metrics.pairwise.cosine_similarity(feats_np)
    K_cos /= K_cos.shape[0]
    ev_cos = torch.linalg.eigvalsh(torch.tensor(K_cos))
    entropy_cos = query_diversity.renyi_entropy(ev_cos.unsqueeze(0), 1.0).item()
    expected_cos = np.exp(entropy_cos)
    print(f"Expected Cosine: {expected_cos}")
    assert np.isclose(score_cos, expected_cos), "Cosine Mismatch"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_calculate_vendi_score()
