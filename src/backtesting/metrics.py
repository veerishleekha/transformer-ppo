"""
Performance metrics for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class PerformanceMetrics:
    """
    Calculate portfolio performance metrics.
    """
    
    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """
        Calculate returns from portfolio values.
        
        Args:
            portfolio_values: Array of portfolio values over time
            
        Returns:
            Array of returns
        """
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    @staticmethod
    def total_return(portfolio_values: np.ndarray) -> float:
        """
        Calculate total return.
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Total return as percentage
        """
        return (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    @staticmethod
    def annualized_return(portfolio_values: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.
        
        Args:
            portfolio_values: Array of portfolio values
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Annualized return as percentage
        """
        total_return = portfolio_values[-1] / portfolio_values[0]
        n_periods = len(portfolio_values)
        years = n_periods / periods_per_year
        
        if years > 0:
            annualized = (total_return ** (1 / years) - 1) * 100
        else:
            annualized = 0
        
        return annualized
    
    @staticmethod
    def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility as percentage
        """
        return np.std(returns) * np.sqrt(periods_per_year) * 100
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        if np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0
        
        return sharpe
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Maximum drawdown as percentage
        """
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_dd = np.min(drawdown) * 100
        
        return max_dd
    
    @staticmethod
    def calmar_ratio(
        portfolio_values: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            portfolio_values: Array of portfolio values
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        ann_return = PerformanceMetrics.annualized_return(portfolio_values, periods_per_year)
        max_dd = abs(PerformanceMetrics.max_drawdown(portfolio_values))
        
        if max_dd > 0:
            calmar = ann_return / max_dd
        else:
            calmar = 0
        
        return calmar
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (focuses on downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
            else:
                sortino = 0
        else:
            sortino = 0
        
        return sortino
    
    @staticmethod
    def average_turnover(weights_history: np.ndarray) -> float:
        """
        Calculate average portfolio turnover.
        
        Args:
            weights_history: Array of portfolio weights over time [T, N]
            
        Returns:
            Average turnover
        """
        if len(weights_history) < 2:
            return 0
        
        turnovers = []
        for i in range(1, len(weights_history)):
            turnover = np.sum(np.abs(weights_history[i] - weights_history[i-1]))
            turnovers.append(turnover)
        
        return np.mean(turnovers)
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive return days).
        
        Args:
            returns: Array of returns
            
        Returns:
            Win rate as percentage
        """
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        
        return (winning_days / total_days) * 100 if total_days > 0 else 0
    
    @staticmethod
    def compute_all_metrics(
        portfolio_values: np.ndarray,
        weights_history: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> Dict:
        """
        Compute all performance metrics.
        
        Args:
            portfolio_values: Array of portfolio values
            weights_history: Array of portfolio weights over time
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Dictionary of all metrics
        """
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        
        metrics = {
            'total_return': PerformanceMetrics.total_return(portfolio_values),
            'annualized_return': PerformanceMetrics.annualized_return(portfolio_values, periods_per_year),
            'volatility': PerformanceMetrics.volatility(returns, periods_per_year),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_values),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(portfolio_values, periods_per_year),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            'average_turnover': PerformanceMetrics.average_turnover(weights_history),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'num_periods': len(portfolio_values)
        }
        
        return metrics
