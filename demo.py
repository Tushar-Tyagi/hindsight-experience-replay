#!/usr/bin/env python3
"""
Hindsight Experience Replay (HER) Demo Script
Load trained model and run demonstration
"""

import os
import argparse
import numpy as np
import torch
import gym

from arguments import get_demo_args
from rl_modules.models import Actor, Critic
from mpi_utils.normalizer import Normalizer
from rl_modules.utils import get_obs_dim, get_goal_dim, get_action_dim


def make_env(env_name):
    """Create and configure the environment"""
    env = gym.make(env_name)
    return env


def load_model(model_path, obs_dim, goal_dim, action_dim, hidden_size, cuda=False):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize networks
    actor = Actor(obs_dim + goal_dim, action_dim, hidden_size)
    critic = Critic(obs_dim + goal_dim + action_dim, 1, hidden_size)
    
    # Load state dicts
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    
    if cuda:
        actor = actor.cuda()
        critic = critic.cuda()
    
    actor.eval()
    critic.eval()
    
    return actor, critic, checkpoint['normalizer']


def run_demo(args):
    """Run demonstration with trained model"""
    # Create environment
    env = make_env(args.env_name)
    
    # Get dimensions
    obs_dim = get_obs_dim(env)
    goal_dim = get_goal_dim(env)
    action_dim = get_action_dim(env)
    
    # Load model
    model_path = f'saved_models/{args.env_name}_{args.epoch}.pt'
    if not os.path.exists(model_path):
        model_path = f'saved_models/{args.env_name}_final.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available models:")
        if os.path.exists('saved_models'):
            for file in os.listdir('saved_models'):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        return
    
    print(f"Loading model from {model_path}")
    actor, critic, normalizer = load_model(
        model_path, obs_dim, goal_dim, action_dim, 
        args.hidden_size, args.cuda
    )
    
    # Run episodes
    success_count = 0
    total_rewards = []
    
    for episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_success = 0
        
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")
        print(f"Goal: {obs['desired_goal']}")
        
        for step in range(args.max_episode_steps):
            # Get action from actor
            with torch.no_grad():
                obs_goal = np.concatenate([obs['observation'], obs['desired_goal']])
                obs_goal_norm = normalizer.normalize(obs_goal)
                obs_goal_tensor = torch.FloatTensor(obs_goal_norm).unsqueeze(0)
                if args.cuda:
                    obs_goal_tensor = obs_goal_tensor.cuda()
                
                action = actor(obs_goal_tensor).cpu().numpy().flatten()
            
            # Take action
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            if info.get('is_success', False):
                episode_success = 1
                print(f"  Success at step {step + 1}!")
            
            # Render environment
            if args.render:
                env.render()
            
            obs = next_obs
            
            if done:
                break
        
        success_count += episode_success
        total_rewards.append(episode_reward)
        
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Success: {'Yes' if episode_success else 'No'}")
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / args.n_episodes
    
    print(f"\n=== Demo Summary ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{args.n_episodes})")
    
    env.close()


if __name__ == '__main__':
    args = get_demo_args()
    run_demo(args)
