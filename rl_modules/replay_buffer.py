#!/usr/bin/env python3
"""
Experience Replay Buffer for DDPG
"""

import numpy as np
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, max_size, obs_dim, goal_dim, action_dim):
        """
        Initialize replay buffer
        
        Args:
            max_size (int): Maximum buffer size
            obs_dim (int): Observation dimension
            goal_dim (int): Goal dimension
            action_dim (int): Action dimension
        """
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        
        # Initialize storage arrays
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.goals = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_goals = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, goal, action, reward, next_obs, next_goal, done):
        """
        Add a transition to the buffer
        
        Args:
            obs (np.ndarray): Current observation
            goal (np.ndarray): Current goal
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_obs (np.ndarray): Next observation
            next_goal (np.ndarray): Next goal
            done (bool): Whether episode is done
        """
        self.obs[self.ptr] = obs
        self.goals[self.ptr] = goal
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_goals[self.ptr] = next_goal
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            dict: Batch of transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'obs': self.obs[indices],
            'goals': self.goals[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_obs[indices],
            'next_goals': self.next_goals[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def sample_her_transitions(self, episode_transitions, her_strategy='future', 
                              replay_k=4, future_p=0.8):
        """
        Sample HER transitions from episode
        
        Args:
            episode_transitions (list): List of transitions from an episode
            her_strategy (str): HER strategy ('future', 'final', 'episode', 'random')
            replay_k (int): Number of HER transitions per real transition
            future_p (float): Probability of using future goals
            
        Returns:
            list: List of HER transitions
        """
        her_transitions = []
        
        for t, transition in enumerate(episode_transitions):
            for _ in range(replay_k):
                # Sample new goal
                new_goal = self._sample_goal(episode_transitions, t, her_strategy, future_p)
                
                # Create HER transition
                her_transition = {
                    'obs': transition['obs'].copy(),
                    'goal': new_goal.copy(),
                    'action': transition['action'].copy(),
                    'reward': self._compute_her_reward(transition['next_obs'], new_goal),
                    'next_obs': transition['next_obs'].copy(),
                    'next_goal': new_goal.copy(),
                    'done': transition['done']
                }
                her_transitions.append(her_transition)
        
        return her_transitions
    
    def _sample_goal(self, episode_transitions, t, strategy, future_p):
        """
        Sample a new goal based on strategy
        
        Args:
            episode_transitions (list): Episode transitions
            t (int): Current time step
            strategy (str): HER strategy
            future_p (float): Future probability
            
        Returns:
            np.ndarray: New goal
        """
        if strategy == 'future':
            if random.random() < future_p and t + 1 < len(episode_transitions):
                future_t = random.randint(t + 1, len(episode_transitions) - 1)
                return episode_transitions[future_t]['next_obs']
            else:
                return episode_transitions[-1]['next_obs']
        
        elif strategy == 'final':
            return episode_transitions[-1]['next_obs']
        
        elif strategy == 'episode':
            random_t = random.randint(0, len(episode_transitions) - 1)
            return episode_transitions[random_t]['next_obs']
        
        elif strategy == 'random':
            random_t = random.randint(0, len(episode_transitions) - 1)
            return episode_transitions[random_t]['next_obs']
        
        else:
            raise ValueError(f"Unknown HER strategy: {strategy}")
    
    def _compute_her_reward(self, achieved_goal, desired_goal, threshold=0.05):
        """
        Compute HER reward
        
        Args:
            achieved_goal (np.ndarray): Achieved goal
            desired_goal (np.ndarray): Desired goal
            threshold (float): Success threshold
            
        Returns:
            float: HER reward
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < threshold else 0.0
    
    def __len__(self):
        """Return current buffer size"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
    
    def get_stats(self):
        """
        Get buffer statistics
        
        Returns:
            dict: Buffer statistics
        """
        return {
            'size': self.size,
            'max_size': self.max_size,
            'ptr': self.ptr,
            'obs_dim': self.obs_dim,
            'goal_dim': self.goal_dim,
            'action_dim': self.action_dim
        }
