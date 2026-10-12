import numpy as np
from pathlib import Path

SEED = 42

# make path relative to constants.py
FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
DEFAULT_TRAINING_DATA_DIR = REPO_ROOT / "training_data"
NEURAL_NET_DIR = REPO_ROOT / "neuralnet"
OUTPUT_DIR = NEURAL_NET_DIR / "output"

SOLUTION_DATA_FILENAME = "solution.bin"
METADATA_FILENAME = "metadata.json"

CSV_FILENAME_FORMAT = "timestep_{:05d}.csv"

CONFIG_KEY = "config"
SOLVER_KEY = "solver"

DOMAIN_LENGTH_KEY = "domain_length"

SPATIAL_STEP_SIZE_KEY = "spatial_step_size"
NUM_DOMAIN_POINTS_KEY = "num_domain_points"
TIME_STEPS_KEY = "time_steps"
TIME_STEP_SIZE_KEY = "time_step_size"
KINEMATIC_VISCOSITY = "kinematic_viscosity"
MAX_U = "max_u"

FEATURE_MATRICES_PATH = OUTPUT_DIR / "feature_matrices.npz"

BIAS_KEY = "bias"
TERMS_KEY = "terms"
AMPLITUDE_KEY = "amplitude"
FREQUENCY_KEY = "frequency"
PHASE_SHIFT_KEY = "phase_shift"

# from solver for dt
ALPHA = 0.4
BETA = 0.2

# how much coarser our nn solve will be, for creating dt from the original solve dt
# TODO make this a range so we don't rely on a single multiplier
COARSENESS_MULTIPLIERS = list(range(1, 65))  # 1 to 64 inclusive
CQ_DENOMINATOR_EPSILON = 1e-8
CQ_MAX_MAGNITUDE = 5.0
# What to do with training points whose reverse-engineered cq is negative.
# cq is a smoothing amount, so a negative value (anti-smoothing) is never a valid target.
#   "drop": remove those points from the training data
#   "clip": keep them with a target of 0 ("add no extra smoothing here")
CQ_NEGATIVE_HANDLING = "drop"
# how many extra features to have around the center
# so like if 3 we will have our center and 3 more in each direction
U_RADIUS = 4

# Caps used for random sampling during feature matrix generation
MAX_TIMESTEPS_PER_SOLUTION = 2000
MAX_SPATIAL_POINTS_PER_TIMESTEP = 500

EPOCHS = 300
CPU_BATCH_SIZE = 2048
GPU_BATCH_SIZE = 65536
