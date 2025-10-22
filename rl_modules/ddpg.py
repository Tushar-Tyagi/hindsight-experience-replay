#!/usr/bin/env python3
"""
Deep Deterministic Policy Gradient (DDPG) implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm
    """
    
    def __init__(self, actor, critic, lr_actor=0.001, lr_critic=0.001, 
                 gamma=0.98, tau=0.05):
        """
        Initialize DDPG
        
        Args:
            actor: Actor network
            critic: Critic network
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            gamma (float): Discount factor
            tau (float): Soft update parameter
        """
        self.actor = actor
        self.critic = critic
        
        # Create target networks
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        
        # Set target networks to eval mode
        self.actor_target.eval()
        self.critic_target.eval()
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # Loss function
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state, add_noise=False, noise_scale=0.1):
        """
        Select action using actor network
        
        Args:
            state (torch.Tensor): State tensor
            add_noise (bool): Whether to add exploration noise
            noise_scale (float): Scale of exploration noise
            
        Returns:
            torch.Tensor: Action tensor
        """
        with torch.no_grad():
            action = self.actor(state)
            
            if add_noise:
                noise = torch.randn_like(action) * noise_scale
                action = torch.clamp(action + noise, -1, 1)
            
            return action
    
    def update(self, replay_buffer, batch_size=256):
        """
        Update actor and critic networks
        
        Args:
            replay_buffer: Replay buffer containing transitions
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training losses
        """
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1)
        next_states = torch.FloatTensor(batch['next_states'])
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1)
        
        # Move to device if needed
        if next(states.parameters()).is_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = self.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def soft_update(self):
        """
        Soft update of target networks
        """
        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """
        Save model checkpoints
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """
        Load model checkpoints
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    def eval(self):
        """Set networks to evaluation mode"""
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
    
    def train(self):
        """Set networks to training mode"""
        self.actor.train()
        self.critic.train()
        self.actor_target.eval()  # Target networks always in eval mode
        self.critic_target.eval()
