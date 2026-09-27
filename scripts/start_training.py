"""
Start the training process for the neural network model.
Usage:
    python3 scripts/start_training.py
"""

import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = "lib/python"


subprocess.run(
    ["python3", "neuralnet/generate_feature_matrices.py"],
    env=env,
    check=True,
)


subprocess.run(
    ["python3", "neuralnet/train.py"],
    env=env,
    check=True,
)
