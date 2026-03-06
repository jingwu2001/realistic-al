import numpy as np
from pathlib import Path

# med

# imb med 10.0 : 2 consecutive 0's
path = Path('experiments/activelearning/cifar10_imb/active-cifar10_low/basic_model-resnet_drop-0.5_aug-cifar_randaugmentMC_acq-bandit_norm-minmax_kernel-rbf_gamma-0.1_ep-200__wloss-True')
for run_dir in path.iterdir():
    for i in range(10):
        extra_info = run_dir / f'extra_info_{i}.npz'
        assert extra_info.exists()
        data = np.load(extra_info)
        extra_info = data['bandit_arm']
        print(extra_info)
    print("\n")