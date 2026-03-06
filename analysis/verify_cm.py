import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns

# Add src and project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set env vars for testing
os.environ['EXPERIMENT_ROOT'] = '/home/jing/Desktop/realistic-al/experiments'
os.environ['DATA_ROOT'] = '/home/jing/Desktop/realistic-al/datasets'

from models.bayesian import BayesianModule
from run_training import get_torchvision_dm
from analysis.confusion import get_confusion_matrix

def cm_for_experiment(experiment_dir: str):
    """
    Get the confusion matrix for a specific experiment.
    
    Args:
        experiment_dir (str): Path to the experiment directory.
    """
    # Load config
    config_path = os.path.join(experiment_dir, ".hydra", "config.yaml")
    print(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    # Ensure data root is correct
    if os.getenv("DATA_ROOT"):
        cfg.trainer.data_root = os.getenv("DATA_ROOT")
    
    # Load model
    print("Initializing model...")
    model = BayesianModule(cfg)
    
    # Find checkpoint
    ckpt_dir = os.path.join(experiment_dir, "loop-9", "checkpoints")
    if not os.path.exists(ckpt_dir):
        # Fallback to look for any checkpoints if loop-9 doesn't exist
        print(f"Warning: {ckpt_dir} not found. Searching recursively...")
        for root, dirs, files in os.walk(experiment_dir):
            if "checkpoints" in dirs:
                ckpt_dir = os.path.join(root, "checkpoints")
                break
                
    if not os.path.exists(ckpt_dir):
         print(f"Error: No checkpoints directory found in {experiment_dir}")
         return

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpt_files:
        print(f"Error: No checkpoint found in {ckpt_dir}")
        return
        
    # Prefer 'last.ckpt' or 'best.ckpt' if available, otherwise take the first one
    if "last.ckpt" in ckpt_files:
        ckpt_name = "last.ckpt"
    elif "best.ckpt" in ckpt_files:
         ckpt_name = "best.ckpt"
    else:
        ckpt_name = ckpt_files[0]
        
    checkpoint_path = os.path.join(ckpt_dir, ckpt_name)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model = model.cuda()
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Load datamodule
    print("Loading datamodule...")
    try:
        datamodule = get_torchvision_dm(cfg)
        datamodule.prepare_data()
        datamodule.setup(stage=None)
        test_loader = datamodule.test_dataloader()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate Confusion Matrix
    print("Calculating confusion matrix...")
    try:
        conf_mat = get_confusion_matrix(model, test_loader)
        print("Confusion Matrix calculated successfully.")
        print("Shape:", conf_mat.shape)
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        return
    
    return conf_mat

if __name__ == "__main__":
    # Test with the experiment path mentioned in the notebook
    exp_dir = '../experiments/activelearning/cifar10/active-cifar10_med/basic_model-resnet_drop-0_aug-cifar_randaugmentMC_acq-vendi_ep-200/2026-01-12_14-26-11-010447'
    abs_exp_dir = os.path.abspath(os.path.join(os.getcwd(), exp_dir))
    
    if os.path.exists(abs_exp_dir):
        cm_for_experiment(abs_exp_dir)
    else:
        print(f"Test experiment directory not found: {abs_exp_dir}")
        # Identify valid experiments to test with
        experiments_root = os.path.abspath(os.path.join(os.getcwd(), "../experiments"))
        print(f"Searching in {experiments_root}...")
