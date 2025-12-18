"""
Experience replay buffer for PPO training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class RolloutBuffer:
    """
    Buffer for storing trajectories for PPO training.
    
    Stores states, actions, rewards, values, log probabilities for
    computing PPO loss.
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_stocks: int,
        num_stock_features: int,
        num_market_features: int,
        device: str = 'cpu'
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            num_stocks: Number of stocks
            num_stock_features: Number of features per stock
            num_market_features: Number of market features
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.num_stocks = num_stocks
        self.num_stock_features = num_stock_features
        self.num_market_features = num_market_features
        self.device = device
        
        # Initialize storage
        self.stock_features = np.zeros((buffer_size, num_stocks, num_stock_features), dtype=np.float32)
        self.market_features = np.zeros((buffer_size, num_market_features), dtype=np.float32)
        self.previous_weights = np.zeros((buffer_size, num_stocks), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_stocks), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        stock_features: np.ndarray,
        market_features: np.ndarray,
        previous_weights: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            stock_features: Stock features
            market_features: Market features
            previous_weights: Previous portfolio weights
            action: Action taken (portfolio weights)
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.stock_features[self.pos] = stock_features
        self.market_features[self.pos] = market_features
        self.previous_weights[self.pos] = previous_weights
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer.
        
        Args:
            batch_size: If provided, return random batches
            
        Returns:
            Dictionary of tensors
        """
        # Determine valid indices
        if self.full:
            indices = np.arange(self.buffer_size)
        else:
            indices = np.arange(self.pos)
        
        if batch_size is not None and batch_size < len(indices):
            indices = np.random.choice(indices, batch_size, replace=False)
        
        return {
            'stock_features': torch.FloatTensor(self.stock_features[indices]).to(self.device),
            'market_features': torch.FloatTensor(self.market_features[indices]).to(self.device),
            'previous_weights': torch.FloatTensor(self.previous_weights[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        # Determine valid length
        if self.full:
            length = self.buffer_size
        else:
            length = self.pos
        
        # Initialize arrays
        returns = np.zeros(length, dtype=np.float32)
        advantages = np.zeros(length, dtype=np.float32)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(length)):
            if t == length - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            
            # GAE
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            
            # Return
            returns[t] = advantages[t] + self.values[t]
        
        return returns, advantages
    
    def clear(self):
        """Clear the buffer."""
        self.pos = 0
        self.full = False
