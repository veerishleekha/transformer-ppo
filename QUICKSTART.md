# Quick Start Guide - Running the Backtest

This guide will help you run the complete Transformer + PPO backtesting notebook and obtain detailed results.

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for neural networks)
- NumPy, Pandas (for data processing)
- Matplotlib, Seaborn (for visualization)
- Gymnasium (for RL environment)
- scikit-learn (for preprocessing)
- PyYAML (for configuration)
- tqdm (for progress bars)

### 2. Verify Installation

```bash
python -c "import torch, pandas, numpy, gymnasium, yaml; print('✓ All dependencies installed')"
```

## Running the Notebook

### Method 1: Jupyter Notebook (Recommended)

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to and open:
# notebooks/transformer_ppo_backtest.ipynb

# Run all cells: Cell → Run All
# Or run cells sequentially: Shift + Enter
```

### Method 2: JupyterLab

```bash
# Start JupyterLab
jupyter lab

# Open: notebooks/transformer_ppo_backtest.ipynb
# Run all cells
```

### Method 3: VS Code

```bash
# Open VS Code
code .

# Install Jupyter extension if not already installed
# Open: notebooks/transformer_ppo_backtest.ipynb
# Click "Run All"
```

## What to Expect

The notebook will:

1. **Load Configuration** (< 1 second)
   - Reads settings from `config/default_config.yaml`

2. **Load & Process Data** (5-10 seconds)
   - Generates synthetic stock data for 20 stocks
   - Computes 16 stock features + 10 market features
   - Preprocesses and normalizes data

3. **Initialize Model** (< 5 seconds)
   - Creates Transformer encoder
   - Initializes PPO agent (~100K parameters)

4. **Train PPO Agent** (5-15 minutes depending on config)
   - Default: 100 iterations
   - Each iteration: 512 environment steps
   - Shows progress bar and periodic updates

5. **Run Backtest** (< 1 minute)
   - Evaluates trained agent on test data
   - Calculates performance metrics

6. **Generate Visualizations** (10-30 seconds)
   - Equity curves
   - Drawdown charts
   - Returns distribution
   - Weight evolution
   - Rolling metrics

7. **Export Results** (< 5 seconds)
   - Saves all results to `results/backtest_YYYYMMDD_HHMMSS/`

**Total Runtime: ~10-20 minutes** (most time spent on training)

## Understanding the Results

After running the notebook, you'll find:

### In the Notebook
- Training progress plots
- Performance metrics tables
- Comparison with benchmark
- Portfolio insights
- Multiple visualizations

### In Results Directory (`results/backtest_YYYYMMDD_HHMMSS/`)

**Start with `SUMMARY.txt`** - Human-readable summary including:
- Performance metrics (Sharpe ratio, returns, drawdown)
- Benchmark comparison
- Top holdings
- Portfolio concentration

**Detailed Files:**
- `performance_metrics.json` - All metrics in JSON format
- `portfolio_timeseries.csv` - Daily portfolio values
- `portfolio_weights.csv` - Daily stock weights
- `portfolio_insights.csv` - Stock-level statistics
- `training_history.csv` - Training episode data
- `config.yaml` - Configuration used

## Customization

### Quick Config Changes

Edit `config/default_config.yaml`:

```yaml
# Reduce training time (faster but less optimal)
training:
  n_iterations: 20  # Default: 100
  
# Increase model capacity (slower but potentially better)
model:
  stock_embedding_dim: 128  # Default: 64
  num_transformer_layers: 4  # Default: 2
  
# Change portfolio constraints
model:
  max_weight: 0.15  # Max 15% per stock (Default: 20%)
  
# Adjust transaction costs
environment:
  transaction_cost: 0.002  # 0.2% (Default: 0.1%)
```

### Using Real Data

In the notebook, change:
```python
# In section "3. Data Loading"
stock_data = loader.load_stock_data(
    use_sample_data=False  # Change to False for real data
)
```

This will use yfinance to download real NIFTY 50 data.

## Troubleshooting

### Issue: "Module not found"
**Solution:** Install requirements: `pip install -r requirements.txt`

### Issue: Training is too slow
**Solution:** Reduce iterations in `config/default_config.yaml`:
```yaml
training:
  n_iterations: 20  # Faster training
  n_steps_per_iteration: 256  # Fewer steps
```

### Issue: Out of memory
**Solution:** Reduce model size:
```yaml
model:
  stock_embedding_dim: 32  # Smaller embedding
  policy_hidden_dim: 32
  value_hidden_dim: 64
```

### Issue: Poor performance
**Solution:** 
1. Increase training iterations
2. Tune hyperparameters (learning rate, entropy coefficient)
3. Try different random seeds

## Next Steps

1. **Analyze Results**: Review the SUMMARY.txt and visualizations
2. **Experiment**: Modify config and re-run to compare
3. **Real Data**: Switch to real market data
4. **Walk-Forward Testing**: Use `BacktestingEngine.walk_forward_backtest()`
5. **Production**: Deploy the trained model for paper trading

## Support

- **Issues**: Open a GitHub issue
- **Documentation**: See README.md
- **Configuration**: See config/default_config.yaml with comments
- **Examples**: See notebooks/transformer_ppo_backtest.ipynb

---

**Ready to Start?**

```bash
jupyter notebook notebooks/transformer_ppo_backtest.ipynb
```

Click "Run All" and wait for the results!
