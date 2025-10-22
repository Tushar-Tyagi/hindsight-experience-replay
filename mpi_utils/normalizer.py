#!/usr/bin/env python3
"""
Observation normalizer for stable training
"""

import numpy as np
from .mpi_utils import mpi_avg, mpi_sum


class Normalizer:
    """
    Online normalizer for observations and goals
    """
    
    def __init__(self, size, eps=1e-8, clip_range=5.0):
        """
        Initialize normalizer
        
        Args:
            size (int): Size of the vector to normalize
            eps (float): Small constant for numerical stability
            clip_range (float): Range for clipping normalized values
        """
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        
        # Running statistics
        self.sum = np.zeros(size, dtype=np.float64)
        self.sum_sq = np.zeros(size, dtype=np.float64)
        self.count = 0
        
        # Normalization parameters
        self.mean = np.zeros(size, dtype=np.float32)
        self.std = np.ones(size, dtype=np.float32)
    
    def update(self, data):
        """
        Update normalizer with new data
        
        Args:
            data (np.ndarray): Data to update with (batch_size, size)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        batch_size = data.shape[0]
        
        # Update running statistics
        self.sum += data.sum(axis=0)
        self.sum_sq += (data ** 2).sum(axis=0)
        self.count += batch_size
        
        # Compute mean and std
        if self.count > 0:
            self.mean = self.sum / self.count
            self.std = np.sqrt(np.maximum(
                self.sum_sq / self.count - self.mean ** 2,
                self.eps
            ))
    
    def normalize(self, data):
        """
        Normalize data
        
        Args:
            data (np.ndarray): Data to normalize
            
        Returns:
            np.ndarray: Normalized data
        """
        normalized = (data - self.mean) / (self.std + self.eps)
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def denormalize(self, data):
        """
        Denormalize data
        
        Args:
            data (np.ndarray): Normalized data
            
        Returns:
            np.ndarray: Denormalized data
        """
        return data * (self.std + self.eps) + self.mean
    
    def get_stats(self):
        """
        Get normalization statistics
        
        Returns:
            dict: Normalization statistics
        """
        return {
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'count': self.count,
            'size': self.size
        }
    
    def set_stats(self, mean, std, count):
        """
        Set normalization statistics
        
        Args:
            mean (np.ndarray): Mean values
            std (np.ndarray): Standard deviation values
            count (int): Number of samples
        """
        self.mean = mean.copy()
        self.std = std.copy()
        self.count = count
        
        # Update running statistics
        self.sum = mean * count
        self.sum_sq = (std ** 2 + mean ** 2) * count
    
    def sync(self):
        """
        Synchronize normalizer across MPI processes
        """
        # Average statistics across all processes
        self.mean = mpi_avg(self.mean)
        self.std = mpi_avg(self.std)
        self.count = int(mpi_avg(np.array([self.count]))[0])
        
        # Update running statistics
        self.sum = self.mean * self.count
        self.sum_sq = (self.std ** 2 + self.mean ** 2) * self.count
    
    def reset(self):
        """Reset normalizer"""
        self.sum = np.zeros(self.size, dtype=np.float64)
        self.sum_sq = np.zeros(self.size, dtype=np.float64)
        self.count = 0
        self.mean = np.zeros(self.size, dtype=np.float32)
        self.std = np.ones(self.size, dtype=np.float32)


class RunningNormalizer:
    """
    Running normalizer with exponential moving average
    """
    
    def __init__(self, size, alpha=0.01, eps=1e-8, clip_range=5.0):
        """
        Initialize running normalizer
        
        Args:
            size (int): Size of the vector to normalize
            alpha (float): Learning rate for exponential moving average
            eps (float): Small constant for numerical stability
            clip_range (float): Range for clipping normalized values
        """
        self.size = size
        self.alpha = alpha
        self.eps = eps
        self.clip_range = clip_range
        
        # Running statistics
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = 0
    
    def update(self, data):
        """
        Update normalizer with new data
        
        Args:
            data (np.ndarray): Data to update with
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        batch_size = data.shape[0]
        
        # Update running statistics
        for i in range(batch_size):
            self.count += 1
            
            # Update mean
            delta = data[i] - self.mean
            self.mean += self.alpha * delta
            
            # Update variance
            delta2 = data[i] - self.mean
            self.var += self.alpha * (delta2 ** 2 - self.var)
    
    def normalize(self, data):
        """
        Normalize data
        
        Args:
            data (np.ndarray): Data to normalize
            
        Returns:
            np.ndarray: Normalized data
        """
        std = np.sqrt(self.var + self.eps)
        normalized = (data - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def denormalize(self, data):
        """
        Denormalize data
        
        Args:
            data (np.ndarray): Normalized data
            
        Returns:
            np.ndarray: Denormalized data
        """
        std = np.sqrt(self.var + self.eps)
        return data * std + self.mean
    
    def get_stats(self):
        """
        Get normalization statistics
        
        Returns:
            dict: Normalization statistics
        """
        return {
            'mean': self.mean.copy(),
            'std': np.sqrt(self.var + self.eps),
            'count': self.count,
            'size': self.size
        }
