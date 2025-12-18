"""
Gym-compatible trading environment for portfolio optimization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
import pandas as pd


class TradingEnvironment(gym.Env):
    """
    Trading environment for portfolio optimization.
    
    State: Stock features, market features, previous weights
    Action: Portfolio weights (continuous, sum to 1)
    Reward: Daily portfolio return - turnover penalty + concentration bonus (optional)
    
    AGGRESSIVE MODE: Set concentration_bonus > 0 to reward concentrated bets
    """
    
    def __init__(
        self,
        stock_sequences: np.ndarray,
        market_sequences: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        transaction_cost: float = 0.001,
        turnover_penalty: float = 0.0005,
        initial_cash: float = 1000000.0,
        normalize_rewards: bool = True,
        random_start: bool = True,
        episode_length: int = 252,  # One year of trading days
        concentration_bonus: float = 0.0,  # Bonus for concentrated portfolios
        momentum_bonus: float = 0.0  # Bonus for momentum-aligned weights
    ):
        """
        Initialize trading environment.
        
        Args:
            stock_sequences: Stock features [T, N, F]
            market_sequences: Market features [T, K]
            returns: Stock returns [T, N]
            dates: Date index
            transaction_cost: Transaction cost as percentage
            turnover_penalty: Penalty for portfolio turnover
            initial_cash: Initial portfolio value
            normalize_rewards: Whether to normalize rewards
            random_start: Whether to randomize episode start date
            episode_length: Length of each episode in days
            concentration_bonus: Reward bonus for concentrated portfolios (HHI)
            momentum_bonus: Reward bonus for aligning weights with recent momentum
        """
        super().__init__()
        
        self.stock_sequences = stock_sequences
        self.market_sequences = market_sequences
        self.returns = returns
        self.dates = dates
        self.transaction_cost = transaction_cost
        self.turnover_penalty = turnover_penalty
        self.initial_cash = initial_cash
        self.normalize_rewards = normalize_rewards
        self.random_start = random_start
        self.episode_length = episode_length
        self.concentration_bonus = concentration_bonus
        self.momentum_bonus = momentum_bonus
        
        self.num_timesteps = len(stock_sequences)
        self.num_stocks = stock_sequences.shape[1]
        self.num_stock_features = stock_sequences.shape[2]
        self.num_market_features = market_sequences.shape[1]
        
        # Define action and observation spaces
        # Action: Portfolio weights (continuous, will be normalized to sum to 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stocks,),
            dtype=np.float32
        )
        
        # Observation: Dict containing stock features, market features, and previous weights
        self.observation_space = spaces.Dict({
            'stock_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_stocks, self.num_stock_features),
                dtype=np.float32
            ),
            'market_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_market_features,),
                dtype=np.float32
            ),
            'previous_weights': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_stocks,),
                dtype=np.float32
            )
        })
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        
        # State variables
        self.current_step = 0
        self.episode_start = 0
        self.previous_weights = None
        self.portfolio_value = initial_cash
        self.cash = initial_cash
        
        # Episode tracking
        self.episode_returns = []
        self.episode_weights = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Randomize start date if enabled
        if self.random_start:
            max_start = self.num_timesteps - self.episode_length - 1
            self.episode_start = np.random.randint(0, max(1, max_start))
        else:
            self.episode_start = 0
        
        self.current_step = self.episode_start
        
        # Initialize with equal weights
        self.previous_weights = np.ones(self.num_stocks) / self.num_stocks
        self.portfolio_value = self.initial_cash
        self.cash = self.initial_cash
        
        # Reset episode tracking
        self.episode_returns = []
        self.episode_weights = []
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Portfolio weights
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Normalize action to ensure it sums to 1
        action = np.array(action, dtype=np.float32)
        action = np.maximum(action, 0)  # Ensure non-negative
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.num_stocks) / self.num_stocks
        
        # Calculate portfolio return
        step_returns = self.returns[self.current_step]
        portfolio_return = np.dot(action, step_returns)
        
        # Calculate turnover
        turnover = np.sum(np.abs(action - self.previous_weights))
        
        # Calculate transaction costs
        transaction_costs = turnover * self.transaction_cost
        
        # Calculate base reward
        reward = portfolio_return - transaction_costs - (turnover * self.turnover_penalty)
        
        # =================================================================
        # AGGRESSIVE MODE BONUSES
        # =================================================================
        
        # Concentration bonus: reward concentrated portfolios
        # HHI = sum of squared weights, ranges from 1/N (equal) to 1 (single stock)
        # Normalize to [0, 1] range: (HHI - 1/N) / (1 - 1/N)
        if self.concentration_bonus > 0:
            hhi = np.sum(action ** 2)
            min_hhi = 1.0 / self.num_stocks
            normalized_hhi = (hhi - min_hhi) / (1.0 - min_hhi + 1e-8)
            reward += self.concentration_bonus * normalized_hhi
        
        # Momentum bonus: reward alignment with recent returns
        # If weight is high on positive return stocks and low on negative return stocks
        if self.momentum_bonus > 0:
            # Use recent returns from features (momentum signal)
            # Rank stocks by return, rank weights - reward correlation
            return_ranks = np.argsort(np.argsort(step_returns))  # 0 = worst, N-1 = best
            weight_ranks = np.argsort(np.argsort(action))
            # Spearman-like correlation bonus
            rank_correlation = np.corrcoef(return_ranks, weight_ranks)[0, 1]
            if not np.isnan(rank_correlation):
                reward += self.momentum_bonus * rank_correlation
        
        # Normalize reward if enabled
        if self.normalize_rewards:
            reward = self._normalize_reward(reward)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return - transaction_costs)
        
        # Update state
        self.previous_weights = action.copy()
        self.current_step += 1
        
        # Track episode data
        self.episode_returns.append(portfolio_return)
        self.episode_weights.append(action)
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.current_step >= min(self.episode_start + self.episode_length, self.num_timesteps - 1):
            truncated = True
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """
        Get current observation.
        
        Returns:
            Dictionary containing state information
        """
        return {
            'stock_features': self.stock_sequences[self.current_step].astype(np.float32),
            'market_features': self.market_sequences[self.current_step].astype(np.float32),
            'previous_weights': self.previous_weights.astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        """
        Get additional info.
        
        Returns:
            Dictionary with additional information
        """
        return {
            'portfolio_value': self.portfolio_value,
            'current_step': self.current_step,
            'date': self.dates[self.current_step] if self.current_step < len(self.dates) else None,
            'episode_length': len(self.episode_returns)
        }
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running statistics.
        
        Args:
            reward: Raw reward
            
        Returns:
            Normalized reward
        """
        self.reward_history.append(reward)
        
        # Update running statistics every 100 steps
        if len(self.reward_history) > 100:
            self.reward_mean = np.mean(self.reward_history[-1000:])
            self.reward_std = np.std(self.reward_history[-1000:]) + 1e-8
        
        normalized = (reward - self.reward_mean) / self.reward_std
        return normalized
    
    def get_episode_statistics(self) -> Dict:
        """
        Get statistics for the completed episode.
        
        Returns:
            Dictionary with episode statistics
        """
        if len(self.episode_returns) == 0:
            return {}
        
        returns_array = np.array(self.episode_returns)
        
        total_return = np.prod(1 + returns_array) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_array)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate average turnover
        if len(self.episode_weights) > 1:
            turnovers = [
                np.sum(np.abs(self.episode_weights[i] - self.episode_weights[i-1]))
                for i in range(1, len(self.episode_weights))
            ]
            avg_turnover = np.mean(turnovers)
        else:
            avg_turnover = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'num_days': len(returns_array)
        }
