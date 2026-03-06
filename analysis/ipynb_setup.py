"""Allow jupyter notebooks in this folder to access data from src
"""
import os
import sys
from pathlib import Path


src_path = Path(os.getcwd()).resolve().parent / "src"

sys.path.append(str(src_path))

# Load .env file
project_root = src_path.parent
env_path = project_root / ".env"
if env_path.exists():
    print(f"Loading .env from {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
else:
    print(f".env not found at {env_path}")
