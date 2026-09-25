"""
Install every dependency the project needs into lib/.
Usage:
    python3 scripts/setup_deps.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "lib"


def install(target, *args, index=None):
    cmd = [sys.executable, "-m", "pip", "install",
           "--target", str(target), *args]
    if index:
        cmd.extend(index)
    subprocess.run(cmd, check=True)


cpp_torch = LIB / "cpp" / "python"
python_deps = LIB / "python"
cpp_torch.mkdir(parents=True, exist_ok=True)
python_deps.mkdir(parents=True, exist_ok=True)


# Currently using CPU torch for everyting. Can switch to GPU torch for training.
TORCH_VERSION = "2.14.0"
INDEX = [
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--extra-index-url", "https://pypi.org/simple",
]
install(cpp_torch, f"torch=={TORCH_VERSION}", index=INDEX)

install(python_deps, "-r", str(ROOT / "requirements.txt"))

print(f"done:\n  C++ torch: {cpp_torch}/torch\n  python deps: {python_deps}")
