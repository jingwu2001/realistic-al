import argparse
import sys
from pathlib import Path
import numpy as np
from pprint import pprint

def read_extra_info(exp_path_str: str):
    """
    Reads extra_info_*.npz files from all runs in the given experiment path.
    Returns a dictionary structured as:
    {
        "run_id": {
            loop_index (int): { dictionary of extra_info content }
        }
    }
    """
    exp_path = Path(exp_path_str)
    if not exp_path.exists():
        print(f"Error: Path {exp_path} does not exist.")
        return

    print(f"Reading from: {exp_path}")
    
    all_runs_data = {}

    # Iterate over subdirectories (runs/seeds)
    for run_dir in sorted(exp_path.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        # print(f"Processing Run: {run_id}")
        
        run_data = {}
        
        # Find all extra_info_*.npz files
        extra_files = list(run_dir.glob("extra_info_*.npz"))
        if not extra_files:
            # print(f"  No extra_info files found in {run_id}")
            continue
            
        # Sort by loop index (the number in the filename)
        # Filename format: extra_info_{i}.npz
        extra_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        for file in extra_files:
            try:
                loop_idx = int(file.stem.split('_')[-1])
                # Load the npz file
                # allow_pickle=True is often needed for dictionaries
                loaded = np.load(file, allow_pickle=True)
                
                # Convert NpzFile to a standard dictionary
                # unpacking items() works, or dict(loaded)
                data_dict = {k: v for k, v in loaded.items()}
                
                # If values are 0-d arrays (scalars), convert them for cleaner reading
                final_dict = {}
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray) and v.ndim == 0:
                        final_dict[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        # Keep arrays as arrays, or convert to list if preferred
                        final_dict[k] = v
                    else:
                        final_dict[k] = v
                
                run_data[loop_idx] = final_dict
                
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        if run_data:
            all_runs_data[run_id] = run_data

    return all_runs_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/read_extra_info.py <path_to_experiment>")
        sys.exit(1)
        
    path_arg = sys.argv[1]
    results = read_extra_info(path_arg)
    
    # Display the organization
    print(f"\nFound data for {len(results)} runs.")
    for run_id, loops in results.items():
        print(f"\nRun: {run_id} ({len(loops)} loops)")
        for loop_idx, data in loops.items():
            for k, v in data.items():
                data[k] = v[[0, 2, 1]]
            print(f"  Loop {loop_idx}: items={list(data.items())}")
            # Uncomment to print full data:
            # pprint(data)
