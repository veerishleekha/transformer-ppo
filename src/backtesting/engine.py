"""
Backtesting engine for walk-forward validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..models.ppo_agent import PPOAgent
from .metrics import PerformanceMetrics


class BacktestingEngine:
    """
    Walk-forward backtesting engine.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        transaction_cost: float = 0.001,
        initial_cash: float = 1000000.0
    ):
        """
        Initialize backtesting engine.
        
        Args:
            agent: Trained PPO agent
            transaction_cost: Transaction cost as percentage
            initial_cash: Initial portfolio value
        """
        self.agent = agent
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash
        
        # Results storage
        self.portfolio_values = []
        self.weights_history = []
        self.returns_history = []
        self.dates_history = []
    
    def run_backtest(
        self,
        stock_sequences: np.ndarray,
        market_sequences: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        deterministic: bool = True
    ) -> Dict:
        """
        Run backtest on test data.
        
        Args:
            stock_sequences: Stock features [T, N, F]
            market_sequences: Market features [T, K]
            returns: Stock returns [T, N]
            dates: Date index
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with backtest results
        """
        num_timesteps = len(stock_sequences)
        num_stocks = stock_sequences.shape[1]
        
        # Initialize
        portfolio_value = self.initial_cash
        previous_weights = np.ones(num_stocks) / num_stocks
        
        portfolio_values = [portfolio_value]
        weights_history = [previous_weights.copy()]
        returns_history = []
        dates_history = [dates[0]]
        
        print("Running backtest...")
        
        for t in tqdm(range(num_timesteps)):
            # Get current state
            stock_features = stock_sequences[t]
            market_features = market_sequences[t]
            
            # Get action from agent
            weights, _ = self.agent.get_action(
                stock_features,
                market_features,
                previous_weights,
                deterministic=deterministic
            )
            
            # Calculate portfolio return
            step_returns = returns[t]
            portfolio_return = np.dot(weights, step_returns)
            
            # Calculate turnover and costs
            turnover = np.sum(np.abs(weights - previous_weights))
            transaction_costs = turnover * self.transaction_cost
            
            # Update portfolio value
            net_return = portfolio_return - transaction_costs
            portfolio_value *= (1 + net_return)
            
            # Store results
            portfolio_values.append(portfolio_value)
            weights_history.append(weights.copy())
            returns_history.append(portfolio_return)
            
            if t < len(dates) - 1:
                dates_history.append(dates[t + 1])
            
            # Update previous weights
            previous_weights = weights.copy()
        
        # Store results
        self.portfolio_values = np.array(portfolio_values)
        self.weights_history = np.array(weights_history)
        self.returns_history = np.array(returns_history)
        self.dates_history = dates_history
        
        # Calculate metrics
        metrics = PerformanceMetrics.compute_all_metrics(
            self.portfolio_values,
            self.weights_history
        )
        
        return {
            'metrics': metrics,
            'portfolio_values': self.portfolio_values,
            'weights_history': self.weights_history,
            'returns_history': self.returns_history,
            'dates': self.dates_history
        }
    
    def walk_forward_backtest(
        self,
        stock_sequences: np.ndarray,
        market_sequences: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        train_window: int = 504,  # 2 years
        test_window: int = 126,   # 6 months
        retrain_frequency: int = 63  # Retrain every quarter
    ) -> Dict:
        """
        Run walk-forward backtesting with periodic retraining.
        
        Args:
            stock_sequences: Stock features
            market_sequences: Market features
            returns: Stock returns
            dates: Date index
            train_window: Training window size
            test_window: Test window size
            retrain_frequency: How often to retrain (in days)
            
        Returns:
            Dictionary with walk-forward results
        """
        num_timesteps = len(stock_sequences)
        
        all_portfolio_values = []
        all_weights = []
        all_returns = []
        all_dates = []
        
        print("Running walk-forward backtest...")
        
        start_idx = 0
        while start_idx + train_window + test_window < num_timesteps:
            train_end = start_idx + train_window
            test_end = min(train_end + test_window, num_timesteps)
            
            print(f"\nWindow: Train {dates[start_idx]} to {dates[train_end-1]}")
            print(f"        Test {dates[train_end]} to {dates[test_end-1]}")
            
            # Get test data
            test_stock = stock_sequences[train_end:test_end]
            test_market = market_sequences[train_end:test_end]
            test_returns = returns[train_end:test_end]
            test_dates = dates[train_end:test_end]
            
            # Run backtest on this window
            results = self.run_backtest(
                test_stock,
                test_market,
                test_returns,
                test_dates,
                deterministic=True
            )
            
            # Aggregate results
            if len(all_portfolio_values) == 0:
                all_portfolio_values.extend(results['portfolio_values'])
            else:
                # Adjust for continuity
                scale = all_portfolio_values[-1] / results['portfolio_values'][0]
                all_portfolio_values.extend(results['portfolio_values'][1:] * scale)
            
            all_weights.extend(results['weights_history'][1:])
            all_returns.extend(results['returns_history'])
            all_dates.extend(results['dates'][1:])
            
            # Move to next window
            start_idx += retrain_frequency
        
        # Calculate overall metrics
        portfolio_values = np.array(all_portfolio_values)
        weights_history = np.array(all_weights)
        
        metrics = PerformanceMetrics.compute_all_metrics(
            portfolio_values,
            weights_history
        )
        
        return {
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'returns_history': np.array(all_returns),
            'dates': all_dates
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get backtest results as DataFrame.
        
        Returns:
            DataFrame with backtest results
        """
        # portfolio_values has T+1 values (including initial value)
        # dates_history has T values (trading days)
        # We need to handle this alignment
        
        if len(self.portfolio_values) == len(self.dates_history) + 1:
            # Create a date for the initial value (day before first trading day)
            if len(self.dates_history) > 0:
                start_date = self.dates_history[0] - pd.Timedelta(days=1)
                all_dates = [start_date] + list(self.dates_history)
            else:
                all_dates = list(self.dates_history)
            
            df = pd.DataFrame({
                'Date': all_dates,
                'Portfolio_Value': self.portfolio_values,
            })
        else:
            # Lengths match, use as-is
            df = pd.DataFrame({
                'Date': self.dates_history,
                'Portfolio_Value': self.portfolio_values[:len(self.dates_history)],
            })
        
        if len(self.returns_history) > 0:
            # Returns start from the first trading day
            df['Returns'] = [np.nan] + list(self.returns_history)
        
        return df
    
    def compare_with_benchmark(
        self,
        benchmark_returns: np.ndarray
    ) -> Dict:
        """
        Compare portfolio with benchmark.
        
        Args:
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with comparison metrics
        """
        portfolio_returns = self.returns_history
        
        # Ensure same length
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate cumulative returns
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)
        
        # Calculate metrics
        portfolio_metrics = PerformanceMetrics.compute_all_metrics(
            portfolio_cumulative,
            self.weights_history[:min_len]
        )
        
        benchmark_metrics = {
            'total_return': PerformanceMetrics.total_return(benchmark_cumulative),
            'annualized_return': PerformanceMetrics.annualized_return(benchmark_cumulative),
            'volatility': PerformanceMetrics.volatility(benchmark_returns),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(benchmark_returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(benchmark_cumulative)
        }
        
        # Calculate tracking error and information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252) * 100
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'excess_return': portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return']
        }
