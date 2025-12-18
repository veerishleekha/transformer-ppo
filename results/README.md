# Results Directory

This directory stores the output from backtest runs.

## Structure

Each backtest run creates a timestamped subdirectory with the following files:

### Performance Metrics
- **`performance_metrics.json`** - Complete performance metrics in JSON format
- **`benchmark_comparison.json`** - Comparison with equal-weight benchmark
- **`SUMMARY.txt`** - Human-readable summary report

### Time Series Data
- **`portfolio_timeseries.csv`** - Daily portfolio values and returns
- **`portfolio_weights.csv`** - Daily portfolio weights for each stock
- **`training_history.csv`** - Training episode rewards and lengths

### Portfolio Analysis
- **`portfolio_insights.csv`** - Stock-level statistics (avg/min/max weights)

### Configuration
- **`config.yaml`** - Configuration used for this run

## Usage

After running the notebook `notebooks/transformer_ppo_backtest.ipynb`, check the most recent timestamped directory for results.

Example:
```
results/
├── backtest_20231218_143022/
│   ├── SUMMARY.txt                    # Start here!
│   ├── performance_metrics.json
│   ├── benchmark_comparison.json
│   ├── portfolio_timeseries.csv
│   ├── portfolio_weights.csv
│   ├── portfolio_insights.csv
│   ├── training_history.csv
│   └── config.yaml
└── README.md
```

## Quick Start

1. Run the complete notebook: `jupyter notebook notebooks/transformer_ppo_backtest.ipynb`
2. Check the latest results directory
3. Open `SUMMARY.txt` for a quick overview
4. Load CSV files for detailed analysis

## Analysis Examples

### Loading Results in Python
```python
import pandas as pd
import json

# Load metrics
with open('results/backtest_20231218_143022/performance_metrics.json') as f:
    metrics = json.load(f)

# Load portfolio timeseries
portfolio_df = pd.read_csv('results/backtest_20231218_143022/portfolio_timeseries.csv')

# Load weights
weights_df = pd.read_csv('results/backtest_20231218_143022/portfolio_weights.csv', index_col=0)
```

### Loading Results in R
```r
library(jsonlite)
library(readr)

# Load metrics
metrics <- fromJSON("results/backtest_20231218_143022/performance_metrics.json")

# Load portfolio timeseries
portfolio_df <- read_csv("results/backtest_20231218_143022/portfolio_timeseries.csv")

# Load weights
weights_df <- read_csv("results/backtest_20231218_143022/portfolio_weights.csv")
```
