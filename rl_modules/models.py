#!/usr/bin/env python3
"""
Neural network models for DDPG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network for DDPG
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        """
        Initialize Actor network
        
        Args:
            state_dim (int): State dimension (observation + goal)
            action_dim (int): Action dimension
            hidden_size (int): Hidden layer size
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state (torch.Tensor): State tensor (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Action tensor (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    """
    Critic network for DDPG
    """
    
    def __init__(self, state_dim, action_dim, output_dim=1, hidden_size=256):
        """
        Initialize Critic network
        
        Args:
            state_dim (int): State dimension (observation + goal)
            action_dim (int): Action dimension
            output_dim (int): Output dimension (default: 1 for Q-value)
            hidden_size (int): Hidden layer size
        """
        super(Critic, self).__init__()
        
        # First hidden layer: state + action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state, action):
        """
        Forward pass
        
        Args:
            state (torch.Tensor): State tensor (batch_size, state_dim)
            action (torch.Tensor): Action tensor (batch_size, action_dim)
            
        Returns:
            torch.Tensor: Q-value tensor (batch_size, output_dim)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class TargetNetwork:
    """
    Target network wrapper for soft updates
    """
    
    def __init__(self, network, tau=0.05):
        """
        Initialize target network
        
        Args:
            network: The network to create a target copy of
            tau (float): Soft update parameter
        """
        self.network = network
        self.target_network = type(network)(*network._get_constructor_args())
        self.tau = tau
        
        # Copy weights
        self.target_network.load_state_dict(self.network.state_dict())
    
    def soft_update(self):
        """Perform soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def __call__(self, *args, **kwargs):
        """Forward pass through target network"""
        return self.target_network(*args, **kwargs)
    
    def eval(self):
        """Set target network to evaluation mode"""
        self.target_network.eval()
    
    def train(self):
        """Set target network to training mode"""
        self.target_network.train()


def create_actor_critic(state_dim, action_dim, hidden_size=256):
    """
    Create actor and critic networks
    
    Args:
        state_dim (int): State dimension
        action_dim (int): Action dimension
        hidden_size (int): Hidden layer size
        
    Returns:
        tuple: (actor, critic) networks
    """
    actor = Actor(state_dim, action_dim, hidden_size)
    critic = Critic(state_dim, action_dim, 1, hidden_size)
    
    return actor, critic


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
