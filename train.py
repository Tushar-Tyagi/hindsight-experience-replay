#!/usr/bin/env python3
"""
Hindsight Experience Replay (HER) Training Script
PyTorch implementation for Fetch robotic environments
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
from her_modules.her import HER
from rl_modules.ddpg import DDPG
from rl_modules.replay_buffer import ReplayBuffer
from mpi_utils.mpi_utils import sync_networks, sync_grads
from mpi_utils.normalizer import Normalizer
from rl_modules.models import Actor, Critic
from rl_modules.utils import get_obs_dim, get_goal_dim, get_action_dim


def make_env(env_name):
    """Create and configure the environment"""
    import gym
    env = gym.make(env_name)
    return env


def train_agent(args):
    """Main training function"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = make_env(args.env_name)
    env.seed(args.seed)
    
    # Get dimensions
    obs_dim = get_obs_dim(env)
    goal_dim = get_goal_dim(env)
    action_dim = get_action_dim(env)
    
    # Initialize normalizer
    normalizer = Normalizer(obs_dim + goal_dim)
    
    # Initialize networks
    actor = Actor(obs_dim + goal_dim, action_dim, args.hidden_size)
    critic = Critic(obs_dim + goal_dim + action_dim, 1, args.hidden_size)
    
    if args.cuda:
        actor = actor.cuda()
        critic = critic.cuda()
    
    # Initialize DDPG
    ddpg = DDPG(actor, critic, args.lr_actor, args.lr_critic, args.gamma, args.tau)
    
    # Initialize HER
    her = HER(args.replay_strategy, args.replay_k, args.future_p)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, goal_dim, action_dim)
    
    # Initialize optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr_critic)
    
    # Initialize tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=f'runs/{args.env_name}_{args.seed}')
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    episode_successes = []
    
    for epoch in range(args.n_epochs):
        for cycle in range(args.n_cycles):
            for episode in range(args.n_episodes):
                # Reset environment
                obs = env.reset()
                episode_reward = 0
                episode_success = 0
                episode_transitions = []
                
                for step in range(args.max_episode_steps):
                    # Get action from actor
                    with torch.no_grad():
                        obs_goal = np.concatenate([obs['observation'], obs['desired_goal']])
                        obs_goal_norm = normalizer.normalize(obs_goal)
                        obs_goal_tensor = torch.FloatTensor(obs_goal_norm).unsqueeze(0)
                        if args.cuda:
                            obs_goal_tensor = obs_goal_tensor.cuda()
                        
                        action = actor(obs_goal_tensor).cpu().numpy().flatten()
                        
                        # Add noise for exploration
                        if args.exploration_noise > 0:
                            noise = np.random.normal(0, args.exploration_noise, action_dim)
                            action = np.clip(action + noise, -1, 1)
                    
                    # Take action
                    next_obs, reward, done, info = env.step(action)
                    
                    # Store transition
                    transition = {
                        'obs': obs['observation'],
                        'goal': obs['desired_goal'],
                        'action': action,
                        'reward': reward,
                        'next_obs': next_obs['observation'],
                        'next_goal': next_obs['desired_goal'],
                        'done': done
                    }
                    episode_transitions.append(transition)
                    
                    episode_reward += reward
                    if info.get('is_success', False):
                        episode_success = 1
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                # Add transitions to replay buffer
                for transition in episode_transitions:
                    replay_buffer.add(
                        transition['obs'], transition['goal'], transition['action'],
                        transition['reward'], transition['next_obs'], transition['next_goal'],
                        transition['done']
                    )
                
                episode_rewards.append(episode_reward)
                episode_successes.append(episode_success)
                
                # HER: Add hindsight experiences
                her_transitions = her.sample_her_transitions(episode_transitions)
                for her_transition in her_transitions:
                    replay_buffer.add(
                        her_transition['obs'], her_transition['goal'], her_transition['action'],
                        her_transition['reward'], her_transition['next_obs'], her_transition['next_goal'],
                        her_transition['done']
                    )
                
                total_steps += len(episode_transitions)
            
            # Update networks
            if len(replay_buffer) > args.batch_size:
                for _ in range(args.n_batches):
                    # Sample batch
                    batch = replay_buffer.sample(args.batch_size)
                    
                    # Normalize observations and goals
                    obs_goals = np.concatenate([batch['obs'], batch['goals']], axis=1)
                    next_obs_goals = np.concatenate([batch['next_obs'], batch['next_goals']], axis=1)
                    
                    obs_goals_norm = normalizer.normalize(obs_goals)
                    next_obs_goals_norm = normalizer.normalize(next_obs_goals)
                    
                    # Convert to tensors
                    obs_goals_tensor = torch.FloatTensor(obs_goals_norm)
                    actions_tensor = torch.FloatTensor(batch['actions'])
                    rewards_tensor = torch.FloatTensor(batch['rewards']).unsqueeze(1)
                    next_obs_goals_tensor = torch.FloatTensor(next_obs_goals_norm)
                    dones_tensor = torch.FloatTensor(batch['dones']).unsqueeze(1)
                    
                    if args.cuda:
                        obs_goals_tensor = obs_goals_tensor.cuda()
                        actions_tensor = actions_tensor.cuda()
                        rewards_tensor = rewards_tensor.cuda()
                        next_obs_goals_tensor = next_obs_goals_tensor.cuda()
                        dones_tensor = dones_tensor.cuda()
                    
                    # Update critic
                    with torch.no_grad():
                        next_actions = actor(next_obs_goals_tensor)
                        next_q_values = critic(next_obs_goals_tensor, next_actions)
                        target_q_values = rewards_tensor + args.gamma * (1 - dones_tensor) * next_q_values
                    
                    current_q_values = critic(obs_goals_tensor, actions_tensor)
                    critic_loss = nn.MSELoss()(current_q_values, target_q_values)
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # Update actor
                    actor_actions = actor(obs_goals_tensor)
                    actor_loss = -critic(obs_goals_tensor, actor_actions).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Soft update target networks
                    ddpg.soft_update()
                
                # Update normalizer
                normalizer.update(obs_goals)
        
        # Logging
        avg_reward = np.mean(episode_rewards[-args.n_episodes:])
        avg_success = np.mean(episode_successes[-args.n_episodes:])
        
        print(f"Epoch {epoch+1}/{args.n_epochs}, "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Success Rate: {avg_success:.2f}, "
              f"Total Steps: {total_steps}")
        
        if args.tensorboard:
            writer.add_scalar('Reward/Episode', avg_reward, epoch)
            writer.add_scalar('Success/Episode', avg_success, epoch)
            writer.add_scalar('Steps/Total', total_steps, epoch)
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs('saved_models', exist_ok=True)
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'normalizer': normalizer,
                'epoch': epoch
            }, f'saved_models/{args.env_name}_{epoch+1}.pt')
    
    # Save final model
    os.makedirs('saved_models', exist_ok=True)
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'normalizer': normalizer,
        'epoch': args.n_epochs
    }, f'saved_models/{args.env_name}_final.pt')
    
    if args.tensorboard:
        writer.close()
    
    env.close()


if __name__ == '__main__':
    args = get_args()
    train_agent(args)
