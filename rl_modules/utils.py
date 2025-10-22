#!/usr/bin/env python3
"""
Utility functions for RL modules
"""

import numpy as np
import gym


def get_obs_dim(env):
    """
    Get observation dimension from environment
    
    Args:
        env: Gym environment
        
    Returns:
        int: Observation dimension
    """
    obs = env.reset()
    if isinstance(obs, dict):
        return obs['observation'].shape[0]
    else:
        return obs.shape[0]


def get_goal_dim(env):
    """
    Get goal dimension from environment
    
    Args:
        env: Gym environment
        
    Returns:
        int: Goal dimension
    """
    obs = env.reset()
    if isinstance(obs, dict):
        return obs['desired_goal'].shape[0]
    else:
        return 0


def get_action_dim(env):
    """
    Get action dimension from environment
    
    Args:
        env: Gym environment
        
    Returns:
        int: Action dimension
    """
    return env.action_space.shape[0]


def get_state_dim(env):
    """
    Get state dimension (observation + goal) from environment
    
    Args:
        env: Gym environment
        
    Returns:
        int: State dimension
    """
    return get_obs_dim(env) + get_goal_dim(env)


def normalize_obs(obs, normalizer):
    """
    Normalize observation using normalizer
    
    Args:
        obs (np.ndarray): Observation
        normalizer: Normalizer object
        
    Returns:
        np.ndarray: Normalized observation
    """
    return normalizer.normalize(obs)


def denormalize_obs(obs, normalizer):
    """
    Denormalize observation using normalizer
    
    Args:
        obs (np.ndarray): Normalized observation
        normalizer: Normalizer object
        
    Returns:
        np.ndarray: Denormalized observation
    """
    return normalizer.denormalize(obs)


def compute_reward(achieved_goal, desired_goal, reward_type='sparse', threshold=0.05):
    """
    Compute reward based on achieved and desired goals
    
    Args:
        achieved_goal (np.ndarray): Achieved goal
        desired_goal (np.ndarray): Desired goal
        reward_type (str): Type of reward ('sparse' or 'dense')
        threshold (float): Success threshold for sparse reward
        
    Returns:
        float: Computed reward
    """
    if reward_type == 'sparse':
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < threshold else 0.0
    elif reward_type == 'dense':
        return -np.linalg.norm(achieved_goal - desired_goal)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def is_goal_achieved(achieved_goal, desired_goal, threshold=0.05):
    """
    Check if goal is achieved
    
    Args:
        achieved_goal (np.ndarray): Achieved goal
        desired_goal (np.ndarray): Desired goal
        threshold (float): Success threshold
        
    Returns:
        bool: True if goal achieved, False otherwise
    """
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return distance < threshold


def get_env_info(env):
    """
    Get environment information
    
    Args:
        env: Gym environment
        
    Returns:
        dict: Environment information
    """
    obs = env.reset()
    
    info = {
        'obs_dim': get_obs_dim(env),
        'goal_dim': get_goal_dim(env),
        'action_dim': get_action_dim(env),
        'state_dim': get_state_dim(env),
        'action_space': env.action_space,
        'observation_space': env.observation_space
    }
    
    return info


def create_env(env_name, seed=None):
    """
    Create and configure environment
    
    Args:
        env_name (str): Environment name
        seed (int): Random seed
        
    Returns:
        gym.Env: Configured environment
    """
    env = gym.make(env_name)
    
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    return env


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed
    """
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
