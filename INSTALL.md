# Installation Guide

This guide will help you install all dependencies for the Hindsight Experience Replay (HER) implementation.

## Prerequisites

- Python 3.7+ (recommended: Python 3.8-3.11)
- pip package manager
- Git (for cloning the repository)

## Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Tushar-Tyagi/hindsight-experience-replay.git
cd hindsight-experience-replay
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n her_env python=3.9
conda activate her_env

# Or using venv
python -m venv her_env
source her_env/bin/activate  # On Windows: her_env\Scripts\activate
```

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt
```

## MuJoCo Installation

MuJoCo is required for the Fetch robotic environments. Here are the installation steps:

### Option 1: Using pip (Recommended for newer systems)

```bash
# Install MuJoCo
pip install mujoco

# Set environment variable (add to your ~/.bashrc or ~/.zshrc)
export MUJOCO_PATH=$(python -c "import mujoco; print(mujoco.__file__.replace('__init__.py', ''))")

# Verify installation
python -c "import mujoco; print('MuJoCo installed successfully')"
```

### Option 2: Manual Installation

1. **Download MuJoCo:**
   - Go to [MuJoCo Downloads](https://mujoco.org/download)
   - Download the latest version for your OS
   - Extract to a directory (e.g., `~/mujoco`)

2. **Set Environment Variables:**
   ```bash
   # Add to your ~/.bashrc or ~/.zshrc
   export MUJOCO_PATH=~/mujoco
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PATH/bin
   export PATH=$PATH:$MUJOCO_PATH/bin
   ```

3. **Install Python bindings:**
   ```bash
   pip install mujoco-py
   ```

### Option 3: Using Conda (Alternative)

```bash
conda install -c conda-forge mujoco
```

## Platform-Specific Instructions

### macOS

```bash
# Install using Homebrew (if you have it)
brew install mujoco

# Or install manually
pip install mujoco
export MUJOCO_PATH=$(python -c "import mujoco; print(mujoco.__file__.replace('__init__.py', ''))")
```

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglfw3 libglfw3-dev libgles2-mesa-dev

# Install MuJoCo
pip install mujoco
export MUJOCO_PATH=$(python -c "import mujoco; print(mujoco.__file__.replace('__init__.py', ''))")
```

### Windows

```bash
# Install MuJoCo
pip install mujoco

# Set environment variable in PowerShell
$env:MUJOCO_PATH = (python -c "import mujoco; print(mujoco.__file__.replace('__init__.py', ''))")

# Or set permanently in System Environment Variables
```

## Optional Dependencies

### MPI for Distributed Training

```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev
pip install mpi4py

# macOS
brew install open-mpi
pip install mpi4py

# Windows (using conda)
conda install -c conda-forge mpi4py
```

### GPU Support (CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

Run the test script to verify your installation:

```bash
python test_her.py
```

Expected output:
```
Running HER implementation tests...

Testing HER Module...
✓ HER module imported successfully
✓ HER generated 12 hindsight transitions

Testing RL Modules...
✓ RL models created successfully
✓ Replay buffer created successfully
✓ Replay buffer operations working

Testing MPI Utils...
✓ Normalizer working correctly

Testing Arguments...
✓ Argument parsing working

Tests completed: 4/4 passed
🎉 All tests passed! HER implementation is ready to use.
```

## Troubleshooting

### Common Issues

1. **MuJoCo Path Error:**
   ```
   RuntimeError: MUJOCO_PATH environment variable is not set
   ```
   **Solution:** Set the `MUJOCO_PATH` environment variable as shown above.

2. **Import Error for gym-robotics:**
   ```
   ImportError: No module named 'gym_robotics'
   ```
   **Solution:** Install gym-robotics: `pip install gym-robotics`

3. **MPI Import Error:**
   ```
   ImportError: No module named 'mpi4py'
   ```
   **Solution:** Install MPI: `pip install mpi4py` or use single-process mode.

4. **CUDA/GPU Issues:**
   - Make sure you have the correct CUDA version installed
   - Install the appropriate PyTorch version for your CUDA version

### Environment Variables

Make sure these are set in your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# MuJoCo
export MUJOCO_PATH=/path/to/mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PATH/bin

# Optional: For better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Development Setup

For development, install additional tools:

```bash
pip install -e .[dev]
```

This installs:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Verify your Python version: `python --version`
3. Check installed packages: `pip list`
4. Run the test script: `python test_her.py`
5. Open an issue on GitHub with your error message and system details

## Next Steps

Once installation is complete:

1. **Run a quick test:**
   ```bash
   python test_her.py
   ```

2. **Start training:**
   ```bash
   python train.py --env-name='FetchReach-v1' --n-epochs=5
   ```

3. **Run experiments:**
   ```bash
   ./run_experiments.sh test
   ```
