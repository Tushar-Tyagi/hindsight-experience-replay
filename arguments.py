#!/usr/bin/env python3
"""
Argument parsing for HER training and demo scripts
"""

import argparse


def get_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser(description='Hindsight Experience Replay (HER) Training')
    
    # Environment arguments
    parser.add_argument('--env-name', type=str, default='FetchReach-v1',
                       help='Environment name (FetchReach-v1, FetchPush-v1, FetchPickAndPlace-v1, FetchSlide-v1)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    
    # Training arguments
    parser.add_argument('--n-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--n-cycles', type=int, default=10,
                       help='Number of cycles per epoch')
    parser.add_argument('--n-episodes', type=int, default=16,
                       help='Number of episodes per cycle')
    parser.add_argument('--n-batches', type=int, default=40,
                       help='Number of batches per cycle')
    parser.add_argument('--max-episode-steps', type=int, default=50,
                       help='Maximum steps per episode')
    
    # HER arguments
    parser.add_argument('--replay-strategy', type=str, default='future',
                       choices=['future', 'final', 'episode', 'random'],
                       help='HER replay strategy')
    parser.add_argument('--replay-k', type=int, default=4,
                       help='Number of HER transitions per real transition')
    parser.add_argument('--future-p', type=float, default=0.8,
                       help='Probability of using future goals in HER')
    
    # DDPG arguments
    parser.add_argument('--lr-actor', type=float, default=0.001,
                       help='Learning rate for actor')
    parser.add_argument('--lr-critic', type=float, default=0.001,
                       help='Learning rate for critic')
    parser.add_argument('--gamma', type=float, default=0.98,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.05,
                       help='Soft update parameter for target networks')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                       help='Replay buffer size')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                       help='Exploration noise for action')
    
    # Network arguments
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size for neural networks')
    
    # Training options
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA for training')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Use TensorBoard for logging')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save model every N epochs')
    
    # MPI arguments
    parser.add_argument('--mpi-rank', type=int, default=0,
                       help='MPI rank')
    parser.add_argument('--mpi-size', type=int, default=1,
                       help='MPI size')
    
    return parser.parse_args()


def get_demo_args():
    """Parse demo arguments"""
    parser = argparse.ArgumentParser(description='Hindsight Experience Replay (HER) Demo')
    
    # Environment arguments
    parser.add_argument('--env-name', type=str, default='FetchReach-v1',
                       help='Environment name')
    parser.add_argument('--epoch', type=int, default=50,
                       help='Model epoch to load')
    
    # Demo arguments
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of demo episodes')
    parser.add_argument('--max-episode-steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during demo')
    
    # Network arguments
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size for neural networks')
    
    # Hardware arguments
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA for inference')
    
    return parser.parse_args()


def get_her_args():
    """Parse HER-specific arguments"""
    parser = argparse.ArgumentParser(description='HER Configuration')
    
    parser.add_argument('--replay-strategy', type=str, default='future',
                       choices=['future', 'final', 'episode', 'random'],
                       help='HER replay strategy')
    parser.add_argument('--replay-k', type=int, default=4,
                       help='Number of HER transitions per real transition')
    parser.add_argument('--future-p', type=float, default=0.8,
                       help='Probability of using future goals in HER')
    
    return parser.parse_args()


def get_ddpg_args():
    """Parse DDPG-specific arguments"""
    parser = argparse.ArgumentParser(description='DDPG Configuration')
    
    parser.add_argument('--lr-actor', type=float, default=0.001,
                       help='Learning rate for actor')
    parser.add_argument('--lr-critic', type=float, default=0.001,
                       help='Learning rate for critic')
    parser.add_argument('--gamma', type=float, default=0.98,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.05,
                       help='Soft update parameter for target networks')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                       help='Replay buffer size')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                       help='Exploration noise for action')
    
    return parser.parse_args()
