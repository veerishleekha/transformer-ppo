# Transformer + PPO Backtesting Engine

A comprehensive Python backtesting engine with modular design for Transformer-based PPO (Proximal Policy Optimization) reinforcement learning system for portfolio optimization targeting NIFTY 50/100 stocks with a long-only, daily trading strategy.

## Overview

This project implements a state-of-the-art portfolio optimization system that combines:
- **Transformer architecture** for learning dynamic correlations between stocks
- **PPO reinforcement learning** for policy optimization
- **Dirichlet distribution** for valid portfolio weight generation
- **Comprehensive backtesting** with walk-forward validation

## Architecture

### Key Components

1. **Stock Feature Encoder (Transformer)**
   - Processes N stocks with F features each
   - Self-attention learns dynamic correlation matrix between stocks
   - Produces contextual embeddings capturing stock relationships

2. **Market Context Encoder**
   - MLP encoder for global market features
   - Includes NIFTY returns, volatility, USD-INR, crude oil, interest rates

3. **Policy Network**
   - Generates portfolio weights using Dirichlet distribution
   - Ensures valid weights (all positive, sum to 1)
   - Includes previous weights to reduce turnover

4. **Value Network (Critic)**
   - Estimates state value for PPO training
   - Uses attention pooling to aggregate stock information

5. **Trading Environment**
   - Gym-compatible environment
   - Realistic transaction costs and turnover penalties
   - Randomized episode starts to prevent memorization

6. **Backtesting Engine**
   - Walk-forward validation
   - Comprehensive performance metrics
   - Rich visualization capabilities

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── features.py          # Feature engineering
│   │   ├── market_data.py       # Market data loading
│   │   └── preprocessing.py     # Data preprocessing
│   ├── models/
│   │   ├── transformer.py       # Transformer encoder
│   │   ├── policy.py            # Policy network with Dirichlet
│   │   ├── critic.py            # Value network
│   │   └── ppo_agent.py         # Combined PPO agent
│   ├── environment/
│   │   └── trading_env.py       # Gym-compatible trading environment
│   ├── training/
│   │   ├── buffer.py            # Experience replay buffer
│   │   └── trainer.py           # PPO training loop
│   └── backtesting/
│       ├── engine.py            # Backtesting engine
│       ├── metrics.py           # Performance metrics
│       └── visualization.py     # Plotting utilities
├── config/
│   └── default_config.yaml      # Configuration file
├── notebooks/
│   └── transformer_ppo_backtest.ipynb  # Main demonstration notebook
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/veerishleekha/strats.git
cd strats

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

**🚀 Want to run a backtest immediately?**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete backtest notebook
jupyter notebook notebooks/transformer_ppo_backtest.ipynb
```

Then click **"Run All"** and wait ~10-20 minutes for complete results!

For detailed instructions, see **[QUICKSTART.md](QUICKSTART.md)**.

### Using the Jupyter Notebook

The easiest way to get started is with the comprehensive demonstration notebook:

```bash
jupyter notebook notebooks/transformer_ppo_backtest.ipynb
```

**The notebook includes:**
- ✅ Complete end-to-end pipeline (data → training → backtesting)
- ✅ Detailed visualizations (equity curves, drawdowns, weight evolution)
- ✅ Performance metrics and benchmark comparison
- ✅ Automatic results export to CSV/JSON files
- ✅ ~18 sections covering all aspects of the system

**After running, you'll get:**
- Training progress plots
- Comprehensive performance metrics (Sharpe, Sortino, Calmar ratios)
- Equity curves vs. benchmark
- Portfolio weight analysis
- All results saved to `results/backtest_YYYYMMDD_HHMMSS/`

See **[QUICKSTART.md](QUICKSTART.md)** for detailed instructions.

### Python API

```python
from src.data.market_data import MarketDataLoader
from src.data.features import FeatureEngineering
from src.data.preprocessing import DataPreprocessor
from src.models.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnvironment
from src.training.trainer import PPOTrainer
from src.backtesting.engine import BacktestingEngine

# Load data
loader = MarketDataLoader()
stock_data = loader.load_stock_data(use_sample_data=True)
nifty_data = loader.load_nifty_index(use_sample_data=True)

# Engineer features
fe = FeatureEngineering()
stock_features = fe.compute_stock_features(stock_data)
market_features = fe.compute_market_features(nifty_data)

# Preprocess
preprocessor = DataPreprocessor()
# ... (see notebook for full example)

# Create agent
agent = PPOAgent(
    num_stock_features=16,
    num_market_features=10,
    num_stocks=20
)

# Train
env = TradingEnvironment(stock_sequences, market_sequences, returns, dates)
trainer = PPOTrainer(agent, env)
trainer.train(n_iterations=100, n_steps_per_iteration=512)

# Backtest
backtester = BacktestingEngine(agent)
results = backtester.run_backtest(test_stock, test_market, test_returns, test_dates)
```

## Configuration

Edit `config/default_config.yaml` to customize:
- Model architecture (embedding dimensions, layers, heads)
- Training hyperparameters (learning rate, PPO parameters)
- Environment settings (transaction costs, penalties)
- Backtesting parameters (windows, retraining frequency)

## Features

### Data Module
- **Feature Engineering**: Price, momentum, volatility, technical indicators
- **Market Context**: NIFTY index, macro features
- **Preprocessing**: Normalization, sequence creation

### Model Module
- **Transformer Encoder**: Multi-head self-attention for stock relationships
- **Dirichlet Policy**: Guarantees valid portfolio weights
- **Attention Pooling**: Sophisticated state value estimation

### Training Module
- **PPO Algorithm**: State-of-the-art policy gradient method
- **GAE**: Generalized Advantage Estimation
- **Experience Replay**: Efficient sample usage

### Backtesting Module
- **Walk-Forward Validation**: Realistic out-of-sample testing
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, drawdown, turnover
- **Rich Visualization**: Equity curves, drawdown charts, weight evolution

## Key Design Considerations

### Preventing Common Failures

| Failure Mode | Solution |
|--------------|----------|
| Overtrading | Include previous weights in state |
| Single stock concentration | Entropy bonus + max weight constraint |
| Memorization | Randomized episode starts |
| Training instability | Reward normalization + gradient clipping |

### Mathematical Formulation

**State**: S_t = {X_t, M_t, W_{t-1}}
- X_t: Stock features [N × F]
- M_t: Market features [K]
- W_{t-1}: Previous weights [N]

**Action**: w_t ~ Dirichlet(α_t)
- Portfolio weights from Dirichlet distribution
- α_t = exp(MLP(E_t, M_embedding, W_{t-1}))

**Reward**: r_t = portfolio_return_t - transaction_costs - turnover_penalty

**Objective**: Maximize expected cumulative reward using PPO

## Performance Metrics

The engine computes:
- Total and annualized returns
- Volatility (annualized)
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Average turnover
- Win rate
- Information ratio (vs benchmark)

## Visualization

Built-in visualization tools:
- Equity curves
- Drawdown charts
- Returns distribution
- Portfolio weight evolution
- Rolling performance metrics
- Performance comparison tables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this work in your research, please cite:

```
@software{transformer_ppo_backtest,
  author = {Your Name},
  title = {Transformer + PPO Backtesting Engine},
  year = {2024},
  url = {https://github.com/veerishleekha/strats}
}
```

## Acknowledgments

- Transformer architecture inspired by "Attention Is All You Need"
- PPO algorithm from "Proximal Policy Optimization Algorithms"
- Portfolio optimization techniques from modern portfolio theory

## Contact

For questions or support, please open an issue on GitHub.
