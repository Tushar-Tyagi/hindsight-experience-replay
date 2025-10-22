#!/usr/bin/env python3
"""
MPI utilities for distributed training
"""

import numpy as np
import torch
from mpi4py import MPI


def mpi_rank():
    """Get MPI rank"""
    return MPI.COMM_WORLD.Get_rank()


def mpi_size():
    """Get MPI size"""
    return MPI.COMM_WORLD.Get_size()


def mpi_avg(x):
    """
    Average a scalar across all MPI processes
    
    Args:
        x: Scalar value to average
        
    Returns:
        float: Averaged value
    """
    if not isinstance(x, (int, float, np.ndarray)):
        return x
    
    if isinstance(x, (int, float)):
        x = np.array([x])
    
    return MPI.COMM_WORLD.allreduce(x, op=MPI.SUM) / mpi_size()


def mpi_sum(x):
    """
    Sum a scalar across all MPI processes
    
    Args:
        x: Scalar value to sum
        
    Returns:
        float: Summed value
    """
    if not isinstance(x, (int, float, np.ndarray)):
        return x
    
    if isinstance(x, (int, float)):
        x = np.array([x])
    
    return MPI.COMM_WORLD.allreduce(x, op=MPI.SUM)


def sync_networks(network):
    """
    Synchronize network parameters across all MPI processes
    
    Args:
        network: PyTorch network to synchronize
    """
    for param in network.parameters():
        param_data = param.data.cpu().numpy()
        MPI.COMM_WORLD.Bcast(param_data, root=0)
        param.data = torch.from_numpy(param_data).to(param.device)


def sync_grads(network):
    """
    Synchronize gradients across all MPI processes
    
    Args:
        network: PyTorch network to synchronize gradients for
    """
    for param in network.parameters():
        if param.grad is not None:
            grad_data = param.grad.data.cpu().numpy()
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, grad_data, op=MPI.SUM)
            param.grad.data = torch.from_numpy(grad_data / mpi_size()).to(param.device)


def sync_params(network):
    """
    Synchronize network parameters using allreduce
    
    Args:
        network: PyTorch network to synchronize
    """
    for param in network.parameters():
        param_data = param.data.cpu().numpy()
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, param_data, op=MPI.SUM)
        param.data = torch.from_numpy(param_data / mpi_size()).to(param.device)


def broadcast_from_root(data, root=0):
    """
    Broadcast data from root process to all other processes
    
    Args:
        data: Data to broadcast
        root (int): Root process rank
        
    Returns:
        Data broadcasted from root
    """
    return MPI.COMM_WORLD.bcast(data, root=root)


def gather_from_all(data, root=0):
    """
    Gather data from all processes to root
    
    Args:
        data: Data to gather
        root (int): Root process rank
        
    Returns:
        list: Gathered data (only on root process)
    """
    return MPI.COMM_WORLD.gather(data, root=root)


def scatter_to_all(data, root=0):
    """
    Scatter data from root to all processes
    
    Args:
        data: Data to scatter (only on root process)
        root (int): Root process rank
        
    Returns:
        Scattered data
    """
    return MPI.COMM_WORLD.scatter(data, root=root)


def mpi_print(*args, **kwargs):
    """
    Print only from rank 0
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments to print
    """
    if mpi_rank() == 0:
        print(*args, **kwargs)


def mpi_log(*args, **kwargs):
    """
    Log only from rank 0
    
    Args:
        *args: Arguments to log
        **kwargs: Keyword arguments to log
    """
    if mpi_rank() == 0:
        print(*args, **kwargs)


def mpi_barrier():
    """Synchronize all processes"""
    MPI.COMM_WORLD.Barrier()


def mpi_abort(exit_code=1):
    """
    Abort all MPI processes
    
    Args:
        exit_code (int): Exit code
    """
    MPI.COMM_WORLD.Abort(exit_code)


def get_rank_info():
    """
    Get MPI rank information
    
    Returns:
        dict: Rank information
    """
    return {
        'rank': mpi_rank(),
        'size': mpi_size(),
        'is_root': mpi_rank() == 0
    }
