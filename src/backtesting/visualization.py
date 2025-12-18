"""
Visualization utilities for backtesting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


class BacktestVisualizer:
    """
    Visualization utilities for backtest results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_equity_curve(
        self,
        portfolio_values: np.ndarray,
        dates: List,
        benchmark_values: Optional[np.ndarray] = None,
        title: str = "Portfolio Equity Curve",
        figsize: Tuple = (14, 6)
    ):
        """
        Plot equity curve.
        
        Args:
            portfolio_values: Portfolio values over time
            dates: Corresponding dates
            benchmark_values: Optional benchmark values
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(dates, portfolio_values, label='Portfolio', linewidth=2)
        
        if benchmark_values is not None:
            ax.plot(dates, benchmark_values, label='Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(
        self,
        portfolio_values: np.ndarray,
        dates: List,
        title: str = "Portfolio Drawdown",
        figsize: Tuple = (14, 6)
    ):
        """
        Plot drawdown chart.
        
        Args:
            portfolio_values: Portfolio values over time
            dates: Corresponding dates
            title: Plot title
            figsize: Figure size
        """
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdown, color='red', linewidth=1.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        title: str = "Returns Distribution",
        figsize: Tuple = (12, 6)
    ):
        """
        Plot returns distribution.
        
        Args:
            returns: Array of returns
            title: Plot title
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(returns) * 100, color='red', linestyle='--', 
                   label=f'Mean: {np.mean(returns)*100:.2f}%')
        ax1.set_xlabel('Daily Returns (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_weight_evolution(
        self,
        weights_history: np.ndarray,
        dates: List,
        stock_names: Optional[List[str]] = None,
        top_n: int = 10,
        title: str = "Portfolio Weight Evolution",
        figsize: Tuple = (14, 8)
    ):
        """
        Plot evolution of portfolio weights over time.
        
        Args:
            weights_history: Array of weights [T, N]
            dates: Corresponding dates
            stock_names: Names of stocks
            top_n: Show top N stocks by average weight
            title: Plot title
            figsize: Figure size
        """
        num_stocks = weights_history.shape[1]
        
        if stock_names is None:
            stock_names = [f'Stock {i+1}' for i in range(num_stocks)]
        
        # Select top N stocks by average weight
        avg_weights = np.mean(weights_history, axis=0)
        top_indices = np.argsort(avg_weights)[-top_n:]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Stacked area plot
        ax.stackplot(
            dates,
            *[weights_history[:, i] for i in top_indices],
            labels=[stock_names[i] for i in top_indices],
            alpha=0.7
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        title: str = "Performance Metrics Comparison",
        figsize: Tuple = (12, 8)
    ):
        """
        Plot comparison of metrics across different strategies.
        
        Args:
            metrics_dict: Dictionary mapping strategy names to metrics
            title: Plot title
            figsize: Figure size
        """
        metric_names = ['annualized_return', 'volatility', 'sharpe_ratio', 
                       'max_drawdown', 'average_turnover']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            
            strategies = list(metrics_dict.keys())
            values = [metrics_dict[s].get(metric, 0) for s in strategies]
            
            bars = ax.bar(strategies, values, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(strategies) > 3:
                ax.set_xticklabels(strategies, rotation=45, ha='right')
            
            # Color negative values differently
            for bar, val in zip(bars, values):
                if val < 0:
                    bar.set_color('red')
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_metrics(
        self,
        returns: np.ndarray,
        dates: List,
        window: int = 63,  # Quarterly
        title: str = "Rolling Performance Metrics",
        figsize: Tuple = (14, 10)
    ):
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Array of returns
            dates: Corresponding dates
            window: Rolling window size
            title: Plot title
            figsize: Figure size
        """
        # Calculate rolling metrics
        rolling_mean = pd.Series(returns).rolling(window).mean() * 252 * 100
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = rolling_mean / rolling_vol
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Rolling return
        axes[0].plot(dates, rolling_mean, linewidth=2)
        axes[0].set_ylabel('Annualized Return (%)', fontsize=11)
        axes[0].set_title('Rolling Return', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Rolling volatility
        axes[1].plot(dates, rolling_vol, linewidth=2, color='orange')
        axes[1].set_ylabel('Annualized Volatility (%)', fontsize=11)
        axes[1].set_title('Rolling Volatility', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Rolling Sharpe
        axes[2].plot(dates, rolling_sharpe, linewidth=2, color='green')
        axes[2].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_performance_report(
        self,
        metrics: Dict,
        figsize: Tuple = (10, 6)
    ):
        """
        Create a visual performance report.
        
        Args:
            metrics: Dictionary of performance metrics
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Prepare text
        report_text = "Portfolio Performance Report\n" + "="*50 + "\n\n"
        
        metric_labels = {
            'total_return': 'Total Return',
            'annualized_return': 'Annualized Return',
            'volatility': 'Volatility (Ann.)',
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio',
            'max_drawdown': 'Max Drawdown',
            'calmar_ratio': 'Calmar Ratio',
            'average_turnover': 'Avg. Turnover',
            'win_rate': 'Win Rate',
            'num_periods': 'Number of Periods'
        }
        
        for key, label in metric_labels.items():
            if key in metrics:
                value = metrics[key]
                if key in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                    report_text += f"{label:.<30} {value:>10.2f}%\n"
                elif key == 'num_periods':
                    report_text += f"{label:.<30} {value:>10.0f}\n"
                else:
                    report_text += f"{label:.<30} {value:>10.4f}\n"
        
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
