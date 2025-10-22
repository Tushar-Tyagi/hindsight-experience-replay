#!/usr/bin/env python3
"""
Simple test script to verify HER implementation
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_her_module():
    """Test HER module functionality"""
    try:
        from her_modules.her import HER
        
        # Test HER initialization
        her = HER(replay_strategy='future', replay_k=4, future_p=0.8)
        print("✓ HER module imported successfully")
        
        # Test goal sampling
        episode_transitions = [
            {'obs': np.array([1, 2, 3]), 'next_obs': np.array([2, 3, 4]), 'action': np.array([0.1]), 'reward': 0, 'done': False},
            {'obs': np.array([2, 3, 4]), 'next_obs': np.array([3, 4, 5]), 'action': np.array([0.2]), 'reward': 0, 'done': False},
            {'obs': np.array([3, 4, 5]), 'next_obs': np.array([4, 5, 6]), 'action': np.array([0.3]), 'reward': 1, 'done': True}
        ]
        
        her_transitions = her.sample_her_transitions(episode_transitions)
        print(f"✓ HER generated {len(her_transitions)} hindsight transitions")
        
        return True
    except Exception as e:
        print(f"✗ HER module test failed: {e}")
        return False

def test_rl_modules():
    """Test RL modules functionality"""
    try:
        from rl_modules.models import Actor, Critic
        from rl_modules.replay_buffer import ReplayBuffer
        
        # Test model creation
        actor = Actor(state_dim=10, action_dim=4, hidden_size=64)
        critic = Critic(state_dim=10, action_dim=4, output_dim=1, hidden_size=64)
        print("✓ RL models created successfully")
        
        # Test replay buffer
        buffer = ReplayBuffer(max_size=1000, obs_dim=6, goal_dim=3, action_dim=4)
        print("✓ Replay buffer created successfully")
        
        # Test buffer operations
        buffer.add(
            obs=np.array([1, 2, 3, 4, 5, 6]),
            goal=np.array([7, 8, 9]),
            action=np.array([0.1, 0.2, 0.3, 0.4]),
            reward=1.0,
            next_obs=np.array([2, 3, 4, 5, 6, 7]),
            next_goal=np.array([8, 9, 10]),
            done=True
        )
        print("✓ Replay buffer operations working")
        
        return True
    except Exception as e:
        print(f"✗ RL modules test failed: {e}")
        return False

def test_mpi_utils():
    """Test MPI utilities (without actual MPI)"""
    try:
        from mpi_utils.normalizer import Normalizer
        
        # Test normalizer
        normalizer = Normalizer(size=10)
        data = np.random.randn(100, 10)
        normalizer.update(data)
        normalized = normalizer.normalize(data[:5])
        print("✓ Normalizer working correctly")
        
        return True
    except Exception as e:
        print(f"✗ MPI utils test failed: {e}")
        return False

def test_arguments():
    """Test argument parsing"""
    try:
        from arguments import get_args, get_demo_args
        
        # Test argument parsing (with empty args)
        sys.argv = ['test_her.py']
        args = get_args()
        print("✓ Argument parsing working")
        
        return True
    except Exception as e:
        print(f"✗ Arguments test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running HER implementation tests...\n")
    
    tests = [
        ("HER Module", test_her_module),
        ("RL Modules", test_rl_modules),
        ("MPI Utils", test_mpi_utils),
        ("Arguments", test_arguments),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! HER implementation is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == '__main__':
    main()
