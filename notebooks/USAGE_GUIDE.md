# Transformer-PPO with Precious Metals - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install yfinance pandas-ta gymnasium torch numpy pandas matplotlib seaborn plotly tqdm scikit-learn
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/transformer_ppo_precious_metals.ipynb
```

### 3. Run the Notebook
Click **"Run All"** or execute cells sequentially.

## What This Notebook Does

### Portfolio Composition
- **15 stocks**: Dynamically selected from NIFTY 50 based on quarterly (3-month) returns
- **2 ETFs**: GOLDBEES (Gold) and SILVERBEES (Silver)
- **Total**: 17 assets optimized using Transformer-PPO

### Key Features

#### 1. Dynamic Stock Selection
- Fetches all NIFTY 50 stocks using yfinance
- Calculates 3-month (63 trading days) returns
- Selects top 15 performing stocks
- No local files needed - all data from Yahoo Finance

#### 2. Precious Metals Integration
- GOLDBEES.NS (Gold ETF)
- SILVERBEES.NS (Silver ETF)
- Provides diversification and inflation hedge

#### 3. Technical Indicators
- **ATR** (Average True Range): Volatility measure
- **MFI** (Money Flow Index): Volume-weighted momentum
- **RSI** (Relative Strength Index): Overbought/oversold
- **Momentum**: 5, 10, 20-day returns
- **Volatility**: Rolling standard deviation

#### 4. Transformer-PPO Model
- Multi-head self-attention for stock relationships
- Dirichlet distribution for valid portfolio weights
- PPO training with 100 episodes
- Comprehensive backtesting

#### 5. Visualizations
- Portfolio value over time
- Drawdown chart
- Monthly returns heatmap
- Weight allocation pie chart
- Weight evolution over time

## Expected Results

After running the notebook, you'll get:

### Performance Metrics
- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Calmar ratio
- Win rate

### Portfolio Allocation
- Final weights for all 17 assets
- Precious metals exposure percentage
- Top 5 holdings
- Weight evolution charts

### Saved Files
- Model checkpoint: `../checkpoints/transformer_ppo_precious_metals.pt`
- Results CSV: `../results/transformer_ppo_precious_metals_results.csv`
- Weights CSV: `../results/final_weights_precious_metals.csv`

## Configuration

You can modify the `CONFIG` dictionary in Section 2:

```python
CONFIG = {
    'num_stocks': 15,           # Number of stocks to select
    'selection_period': 63,      # Quarterly lookback (trading days)
    'n_episodes': 100,          # Training episodes
    'learning_rate': 3e-4,      # PPO learning rate
    'max_weight': 0.25,         # Maximum weight per asset
    # ... and more
}
```

## Execution Time

- **Data download**: ~2-5 minutes (50+ stocks)
- **Feature engineering**: ~1-2 minutes
- **Training**: ~10-15 minutes (100 episodes)
- **Backtesting**: ~1-2 minutes
- **Total**: ~15-25 minutes

## Common Issues

### Issue 1: yfinance download fails
**Solution**: Some tickers may not be available. The notebook handles this gracefully and continues with available data.

### Issue 2: CUDA not available
**Solution**: The notebook will automatically use CPU. Training will be slower but still works.

### Issue 3: ImportError for src modules
**Solution**: Make sure you're running from the notebooks/ directory, or the sys.path.insert() at the top of the notebook will add the src/ directory.

## Understanding the Output

### Final Weights Example
```
Asset           Weight  Weight_Pct
---------------------------------
RELIANCE        0.082   8.20%
GOLDBEES        0.075   7.50%
HDFCBANK        0.068   6.80%
...
SILVERBEES      0.045   4.50%
```

### Precious Metals Allocation
The notebook specifically highlights:
- Total precious metals exposure (GOLDBEES + SILVERBEES)
- Individual allocation to gold and silver
- Evolution of precious metals weights over time

## Next Steps

After running the notebook:

1. **Analyze the results**
   - Check Sharpe ratio (>1.0 is good)
   - Review maximum drawdown (<20% is good)
   - Compare to benchmark returns

2. **Modify parameters**
   - Try different `num_stocks` (10-20)
   - Adjust `max_weight` for more/less concentration
   - Change `selection_period` for different lookback

3. **Experiment with features**
   - Add more technical indicators
   - Include fundamental data
   - Try sector-relative features

4. **Production deployment**
   - Save the trained model
   - Set up periodic retraining
   - Implement live trading interface

## Support

For issues or questions:
1. Check the notebook comments and documentation
2. Review the main README.md
3. Open an issue on GitHub

## Credits

This notebook integrates:
- **Transformer architecture** from existing transformer-PPO codebase
- **yfinance** for data fetching
- **PPO algorithm** for reinforcement learning
- **Technical analysis** indicators for feature engineering

Enjoy optimizing your portfolio! 🚀📈💎
