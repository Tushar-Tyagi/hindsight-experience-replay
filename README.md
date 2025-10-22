# Hindsight Experience Replay (HER)

This is a PyTorch implementation of Hindsight Experience Replay (HER) for goal-conditioned reinforcement learning, specifically designed for Fetch robotic environments.

## Overview

Hindsight Experience Replay is a technique that allows agents to learn from failed attempts by treating achieved goals as desired goals. This is particularly useful in sparse reward environments where the agent rarely receives positive feedback.

## Features

- **Complete HER Implementation**: Full implementation of the HER algorithm with multiple replay strategies
- **DDPG Integration**: Deep Deterministic Policy Gradient (DDPG) as the base RL algorithm
- **Fetch Environments**: Support for all Fetch robotic environments (Reach, Push, PickAndPlace, Slide)
- **Distributed Training**: MPI support for multi-process training
- **Flexible Configuration**: Comprehensive argument parsing and configuration options
- **Visualization**: TensorBoard integration for training monitoring
- **Demo Mode**: Easy-to-use demonstration script for trained models

## Installation

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Tushar-Tyagi/hindsight-experience-replay.git
cd hindsight-experience-replay
```

2. **Create virtual environment (recommended):**
```bash
conda create -n her_env python=3.9
conda activate her_env
# Or using venv: python -m venv her_env && source her_env/bin/activate
```

3. **Install dependencies:**
```bash
# Option 1: Full installation (includes MuJoCo)
pip install -r requirements.txt

# Option 2: Minimal installation (if MuJoCo issues)
pip install -r requirements-minimal.txt
```

4. **Install MuJoCo (required for Fetch environments):**
```bash
# Install MuJoCo
pip install mujoco

# Set environment variable
export MUJOCO_PATH=$(python -c "import mujoco; print(mujoco.__file__.replace('__init__.py', ''))")

# Verify installation
python -c "import mujoco; print('MuJoCo installed successfully')"
```

### Detailed Installation

For detailed installation instructions, including platform-specific setup and troubleshooting, see [INSTALL.md](INSTALL.md).

### Verify Installation

```bash
python test_her.py
```

## Quick Start

### Training

Train on FetchReach-v1 (single process):
```bash
python train.py --env-name='FetchReach-v1' --n-cycles=10
```

Train on FetchPush-v1 (multi-process):
```bash
mpirun -np 8 python train.py --env-name='FetchPush-v1'
```

Train on FetchPickAndPlace-v1 (multi-process):
```bash
mpirun -np 16 python train.py --env-name='FetchPickAndPlace-v1'
```

Train on FetchSlide-v1 (multi-process):
```bash
mpirun -np 8 python train.py --env-name='FetchSlide-v1' --n-epochs=200
```

### Demo

Run demonstration with trained model:
```bash
python demo.py --env-name='FetchReach-v1' --epoch=50 --render
```

## Configuration

### Training Arguments

- `--env-name`: Environment name (FetchReach-v1, FetchPush-v1, FetchPickAndPlace-v1, FetchSlide-v1)
- `--n-epochs`: Number of training epochs (default: 50)
- `--n-cycles`: Number of cycles per epoch (default: 10)
- `--n-episodes`: Number of episodes per cycle (default: 16)
- `--n-batches`: Number of batches per cycle (default: 40)
- `--max-episode-steps`: Maximum steps per episode (default: 50)

### HER Arguments

- `--replay-strategy`: HER replay strategy ('future', 'final', 'episode', 'random')
- `--replay-k`: Number of HER transitions per real transition (default: 4)
- `--future-p`: Probability of using future goals (default: 0.8)

### DDPG Arguments

- `--lr-actor`: Learning rate for actor (default: 0.001)
- `--lr-critic`: Learning rate for critic (default: 0.001)
- `--gamma`: Discount factor (default: 0.98)
- `--tau`: Soft update parameter (default: 0.05)
- `--batch-size`: Batch size for training (default: 256)
- `--buffer-size`: Replay buffer size (default: 1000000)

### Hardware Arguments

- `--cuda`: Use CUDA for training
- `--tensorboard`: Enable TensorBoard logging

## Project Structure

```
HER/
├── train.py                 # Main training script
├── demo.py                  # Demonstration script
├── arguments.py             # Argument parsing
├── requirements.txt         # Dependencies
├── README.md               # This file
├── her_modules/            # HER implementation
│   ├── __init__.py
│   └── her.py
├── rl_modules/             # RL algorithms and utilities
│   ├── __init__.py
│   ├── ddpg.py
│   ├── replay_buffer.py
│   ├── models.py
│   └── utils.py
├── mpi_utils/              # MPI utilities
│   ├── __init__.py
│   ├── mpi_utils.py
│   └── normalizer.py
├── saved_models/           # Trained model checkpoints
├── experiments/            # Experiment results and logs
└── figures/                # Training curves and visualizations
```

## HER Algorithm

The HER algorithm works by:

1. **Collecting Episodes**: The agent interacts with the environment and collects transitions
2. **Goal Relabeling**: Failed episodes are relabeled with achieved goals as desired goals
3. **Experience Replay**: Both original and relabeled transitions are stored in the replay buffer
4. **Training**: The agent learns from both successful and "hindsight" experiences

### Replay Strategies

- **Future**: Sample goals from future transitions in the same episode
- **Final**: Always use the final achieved goal
- **Episode**: Sample goals from any transition in the episode
- **Random**: Sample goals from any transition (including current)

## Results

The implementation achieves competitive performance on Fetch environments:

- **FetchReach-v1**: High success rate with fast convergence
- **FetchPush-v1**: Robust pushing behavior with good generalization
- **FetchPickAndPlace-v1**: Complex manipulation with high success rate
- **FetchSlide-v1**: Challenging sliding task with good performance

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir=runs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Baselines for the original HER implementation
- PyTorch team for the excellent deep learning framework
- MuJoCo team for the physics simulation environment

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{her_pytorch,
  title={Hindsight Experience Replay - PyTorch Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/hindsight-experience-replay}
}
```

## References

- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) - Andrychowicz et al.
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) - Lillicrap et al.
