#!/usr/bin/env python3
"""
Hindsight Experience Replay (HER) implementation
"""

import numpy as np
import random


class HER:
    """
    Hindsight Experience Replay implementation
    
    This class implements the core HER algorithm that generates hindsight
    experiences by relabeling failed episodes with achieved goals.
    """
    
    def __init__(self, replay_strategy='future', replay_k=4, future_p=0.8):
        """
        Initialize HER
        
        Args:
            replay_strategy (str): Strategy for selecting goals ('future', 'final', 'episode', 'random')
            replay_k (int): Number of HER transitions per real transition
            future_p (float): Probability of using future goals (only for 'future' strategy)
        """
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.future_p = future_p
        
        if self.replay_strategy == 'future' and self.future_p is None:
            self.future_p = 0.8
        elif self.replay_strategy == 'future' and self.future_p is None:
            self.future_p = 0.8
    
    def sample_her_transitions(self, episode_transitions):
        """
        Sample hindsight experiences from episode transitions
        
        Args:
            episode_transitions (list): List of transitions from an episode
            
        Returns:
            list: List of HER transitions
        """
        her_transitions = []
        
        for t, transition in enumerate(episode_transitions):
            for _ in range(self.replay_k):
                # Sample a new goal
                new_goal = self._sample_goal(episode_transitions, t)
                
                # Create new transition with relabeled goal
                her_transition = self._relabel_transition(transition, new_goal)
                her_transitions.append(her_transition)
        
        return her_transitions
    
    def _sample_goal(self, episode_transitions, t):
        """
        Sample a new goal based on the replay strategy
        
        Args:
            episode_transitions (list): Episode transitions
            t (int): Current time step
            
        Returns:
            np.ndarray: New goal
        """
        if self.replay_strategy == 'future':
            # Sample future goals with probability future_p
            if random.random() < self.future_p and t + 1 < len(episode_transitions):
                # Sample from future transitions
                future_t = random.randint(t + 1, len(episode_transitions) - 1)
                return episode_transitions[future_t]['next_obs']
            else:
                # Use final goal
                return episode_transitions[-1]['next_obs']
        
        elif self.replay_strategy == 'final':
            # Always use the final achieved goal
            return episode_transitions[-1]['next_obs']
        
        elif self.replay_strategy == 'episode':
            # Sample from any transition in the episode
            random_t = random.randint(0, len(episode_transitions) - 1)
            return episode_transitions[random_t]['next_obs']
        
        elif self.replay_strategy == 'random':
            # Sample from any transition in the episode (including current)
            random_t = random.randint(0, len(episode_transitions) - 1)
            return episode_transitions[random_t]['next_obs']
        
        else:
            raise ValueError(f"Unknown replay strategy: {self.replay_strategy}")
    
    def _relabel_transition(self, transition, new_goal):
        """
        Relabel a transition with a new goal
        
        Args:
            transition (dict): Original transition
            new_goal (np.ndarray): New goal to use
            
        Returns:
            dict: Relabeled transition
        """
        # Compute new reward based on new goal
        new_reward = self._compute_reward(transition['next_obs'], new_goal)
        
        # Create new transition
        her_transition = {
            'obs': transition['obs'].copy(),
            'goal': new_goal.copy(),
            'action': transition['action'].copy(),
            'reward': new_reward,
            'next_obs': transition['next_obs'].copy(),
            'next_goal': new_goal.copy(),
            'done': transition['done']
        }
        
        return her_transition
    
    def _compute_reward(self, achieved_goal, desired_goal, reward_type='sparse'):
        """
        Compute reward based on achieved and desired goals
        
        Args:
            achieved_goal (np.ndarray): Achieved goal
            desired_goal (np.ndarray): Desired goal
            reward_type (str): Type of reward ('sparse' or 'dense')
            
        Returns:
            float: Computed reward
        """
        if reward_type == 'sparse':
            # Sparse reward: 1 if goal achieved, 0 otherwise
            return self._is_goal_achieved(achieved_goal, desired_goal)
        elif reward_type == 'dense':
            # Dense reward: negative distance to goal
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    def _is_goal_achieved(self, achieved_goal, desired_goal, threshold=0.05):
        """
        Check if goal is achieved based on distance threshold
        
        Args:
            achieved_goal (np.ndarray): Achieved goal
            desired_goal (np.ndarray): Desired goal
            threshold (float): Distance threshold for success
            
        Returns:
            float: 1.0 if goal achieved, 0.0 otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < threshold else 0.0
    
    def get_stats(self):
        """
        Get HER statistics
        
        Returns:
            dict: HER statistics
        """
        return {
            'replay_strategy': self.replay_strategy,
            'replay_k': self.replay_k,
            'future_p': self.future_p
        }
