"""
PPO training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .buffer import RolloutBuffer
from ..models.ppo_agent import PPOAgent
from ..environment.trading_env import TradingEnvironment


class PPOTrainer:
    """
    Trainer for PPO agent.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        env: TradingEnvironment,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = 'cpu'
    ):
        """
        Initialize PPO trainer.
        
        Args:
            agent: PPO agent
            env: Trading environment
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Batch size for updates
            buffer_size: Size of rollout buffer
            device: Device to train on
        """
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            num_stocks=env.num_stocks,
            num_stock_features=env.num_stock_features,
            num_market_features=env.num_market_features,
            device=device
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def collect_rollout(self, n_steps: int) -> Dict:
        """
        Collect rollout data.
        
        Args:
            n_steps: Number of steps to collect
            
        Returns:
            Dictionary with rollout statistics
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(n_steps):
            # Get action from agent
            with torch.no_grad():
                weights, log_prob = self.agent.get_action(
                    obs['stock_features'],
                    obs['market_features'],
                    obs['previous_weights'],
                    deterministic=False
                )
                
                # Get value estimate
                stock_feat = torch.FloatTensor(obs['stock_features']).unsqueeze(0).to(self.device)
                market_feat = torch.FloatTensor(obs['market_features']).unsqueeze(0).to(self.device)
                value = self.agent.get_value(stock_feat, market_feat).item()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(weights)
            done = terminated or truncated
            
            # Store in buffer
            self.buffer.add(
                stock_features=obs['stock_features'],
                market_features=obs['market_features'],
                previous_weights=obs['previous_weights'],
                action=weights,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            # Reset if done
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Get last value for GAE
        with torch.no_grad():
            stock_feat = torch.FloatTensor(obs['stock_features']).unsqueeze(0).to(self.device)
            market_feat = torch.FloatTensor(obs['market_features']).unsqueeze(0).to(self.device)
            last_value = self.agent.get_value(stock_feat, market_feat).item()
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        return {
            'returns': returns,
            'advantages': advantages
        }
    
    def update(self, returns: np.ndarray, advantages: np.ndarray):
        """
        Update policy using PPO.
        
        Args:
            returns: Computed returns
            advantages: Computed advantages
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Get data from buffer
        data = self.buffer.get()
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Create random batches
            indices = np.arange(len(returns))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_stock_features = data['stock_features'][batch_indices]
                batch_market_features = data['market_features'][batch_indices]
                batch_previous_weights = data['previous_weights'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['old_log_probs'][batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_stock_features,
                    batch_market_features,
                    batch_previous_weights,
                    batch_actions
                )
                
                # Compute ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy_loss.item())
        
        # Clear buffer
        self.buffer.clear()
    
    def train(self, n_iterations: int, n_steps_per_iteration: int) -> Dict:
        """
        Train the agent.
        
        Args:
            n_iterations: Number of training iterations
            n_steps_per_iteration: Number of steps to collect per iteration
            
        Returns:
            Dictionary with training statistics
        """
        print(f"Starting PPO training for {n_iterations} iterations...")
        
        for iteration in tqdm(range(n_iterations)):
            # Collect rollout
            rollout_data = self.collect_rollout(n_steps_per_iteration)
            
            # Update policy
            self.update(rollout_data['returns'], rollout_data['advantages'])
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
                print(f"\nIteration {iteration + 1}/{n_iterations}")
                print(f"  Avg Episode Reward: {avg_reward:.4f}")
                print(f"  Avg Episode Length: {avg_length:.1f}")
                print(f"  Policy Loss: {np.mean(self.policy_losses[-10:]):.4f}")
                print(f"  Value Loss: {np.mean(self.value_losses[-10:]):.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
