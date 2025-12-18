"""
Smart Stock Selection Module

Implements intelligent stock filtering and selection strategies
to reduce selection bias and improve portfolio quality.

Enhanced with:
- Historical NIFTY 50 composition (bi-annual updates)
- Rolling stock selection to avoid look-ahead bias
- Survivorship bias mitigation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SmartStockSelector:
    """
    Intelligent stock selector that filters stocks based on
    fundamental and technical criteria.
    
    Now includes:
    - Historical NIFTY 50 composition by year
    - Bi-annual (6-month) rolling selection
    - Survivorship bias mitigation
    """
    
    # Historical NIFTY 50 composition by year (based on actual index changes)
    # This prevents survivorship bias by using the composition that was actually
    # available at each point in time
    NIFTY_50_HISTORICAL = {
        2015: [
            'ACC', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO',
            'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL', 'CAIRN',
            'CIPLA', 'COALINDIA', 'DRREDDY', 'GAIL', 'GRASIM',
            'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO',
            'HINDUNILVR', 'ICICIBANK', 'IDFC', 'INDUSINDBK', 'INFY',
            'ITC', 'JINDALSTEL', 'KOTAKBANK', 'LT', 'LUPIN',
            'M&M', 'MARUTI', 'NMDC', 'NTPC', 'ONGC',
            'PNB', 'POWERGRID', 'RELIANCE', 'SBIN', 'SSLT',
            'SUNPHARMA', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TCS',
            'TECHM', 'ULTRACEMCO', 'WIPRO', 'YESBANK', 'ZEEL'
        ],
        2016: [
            'ACC', 'ADANIPORTS', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL',
            'BOSCHLTD', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT',
            'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'IDEA',
            'INDUSINDBK', 'INFY', 'ITC', 'KOTAKBANK', 'LT',
            'LUPIN', 'M&M', 'MARUTI', 'NTPC', 'ONGC',
            'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMOTORS',
            'TATASTEEL', 'TCS', 'TECHM', 'ULTRACEMCO', 'WIPRO',
            'YESBANK', 'ZEEL'
        ],
        2017: [
            'ACC', 'ADANIPORTS', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BANKBARODA', 'BHARTIARTL', 'BPCL',
            'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL',
            'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO',
            'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
            'INFRATEL', 'INFY', 'ITC', 'KOTAKBANK', 'LT',
            'LUPIN', 'M&M', 'MARUTI', 'NTPC', 'ONGC',
            'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMOTORS',
            'TATASTEEL', 'TCS', 'TECHM', 'ULTRACEMCO', 'UPL',
            'VEDL', 'WIPRO', 'YESBANK', 'ZEEL'
        ],
        2018: [
            'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
            'BAJAJFINSV', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA',
            'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM',
            'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO',
            'HINDPETRO', 'HINDUNILVR', 'IBULHSGFIN', 'ICICIBANK', 'INDUSINDBK',
            'INFRATEL', 'INFY', 'IOC', 'ITC', 'JSWSTEEL',
            'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA',
            'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
            'ULTRACEMCO', 'UPL', 'VEDL', 'WIPRO', 'YESBANK', 'ZEEL'
        ],
        2019: [
            'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
            'BAJAJFINSV', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA',
            'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM',
            'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO',
            'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFRATEL', 'INFY',
            'IOC', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT',
            'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC',
            'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMOTORS',
            'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'ULTRACEMCO',
            'UPL', 'VEDL', 'WIPRO', 'YESBANK', 'ZEEL'
        ],
        2020: [
            'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
            'BAJAJFINSV', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA',
            'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GAIL',
            'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
            'INFY', 'IOC', 'ITC', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN',
            'SHREECEM', 'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TCS',
            'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'
        ],
        2021: [
            'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
            'BAJAJFINSV', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA',
            'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM',
            'HCLTECH', 'HDFC', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO',
            'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY',
            'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M',
            'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID',
            'RELIANCE', 'SBILIFE', 'SBIN', 'SHREECEM', 'SUNPHARMA',
            'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM',
            'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM'
        ],
        2022: [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL', 'BPCL',
            'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
            'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK',
            'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
            'INDUSINDBK', 'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN',
            'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS',
            'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'
        ],
        2023: [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL', 'BPCL',
            'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
            'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
            'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT',
            'LTIM', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN',
            'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS',
            'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'
        ],
        2024: [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
            'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
            'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
            'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
            'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
            'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
            'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
            'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM'
        ],
        2025: [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL', 'BEL',
            'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY',
            'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
            'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT',
            'LTIM', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SHRIRAMFIN', 'SBILIFE',
            'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM', 'TATAMOTORS',
            'TATASTEEL', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO'
        ]
    }
    
    # Current NIFTY 50 (fallback)
    NIFTY_50_STOCKS = NIFTY_50_HISTORICAL[2024]
    
    def __init__(
        self,
        min_data_days: int = 500,
        max_volatility_percentile: float = 75,
        min_return_percentile: float = 25,
        min_volume_percentile: float = 25,
        lookback_days: int = 252,
        rebalance_frequency: str = 'biannual'  # 'annual', 'biannual', 'quarterly'
    ):
        """
        Initialize the stock selector.
        
        Args:
            min_data_days: Minimum trading days required
            max_volatility_percentile: Remove stocks above this volatility percentile
            min_return_percentile: Remove stocks below this return percentile
            min_volume_percentile: Remove stocks below this volume percentile
            lookback_days: Days to look back for calculations (default 126 = 6 months)
            rebalance_frequency: How often to rebalance ('annual', 'biannual', 'quarterly')
        """
        self.min_data_days = min_data_days
        self.max_volatility_percentile = max_volatility_percentile
        self.min_return_percentile = min_return_percentile
        self.min_volume_percentile = min_volume_percentile
        self.lookback_days = lookback_days
        self.rebalance_frequency = rebalance_frequency
        
        self.stock_metrics = {}
        self.selection_log = []
        self.all_stock_data = {}  # Cache for loaded data
        self.selection_history = []  # Track selection changes over time
    
    def get_nifty50_for_date(self, date: pd.Timestamp) -> List[str]:
        """
        Get NIFTY 50 composition for a given date.
        Uses historical composition to avoid survivorship bias.
        
        For bi-annual updates:
        - H1 (Jan-Jun): Use previous year's composition
        - H2 (Jul-Dec): Use current year's composition
        """
        year = date.year
        month = date.month
        
        # For H1, use previous year's composition (conservative approach)
        # For H2, use current year's composition
        if month <= 6:
            lookup_year = year - 1
        else:
            lookup_year = year
        
        # Find the closest available year
        available_years = sorted(self.NIFTY_50_HISTORICAL.keys())
        
        if lookup_year in self.NIFTY_50_HISTORICAL:
            return self.NIFTY_50_HISTORICAL[lookup_year]
        elif lookup_year < min(available_years):
            return self.NIFTY_50_HISTORICAL[min(available_years)]
        else:
            return self.NIFTY_50_HISTORICAL[max(available_years)]
    
    def get_rebalance_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """
        Generate rebalancing dates based on frequency.
        
        For bi-annual: Jan 1 and Jul 1 of each year
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        dates = []
        
        if self.rebalance_frequency == 'biannual':
            # Generate Jan 1 and Jul 1 for each year
            current_year = start.year
            while current_year <= end.year:
                jan_date = pd.Timestamp(year=current_year, month=1, day=1)
                jul_date = pd.Timestamp(year=current_year, month=7, day=1)
                
                if start <= jan_date <= end:
                    dates.append(jan_date)
                if start <= jul_date <= end:
                    dates.append(jul_date)
                
                current_year += 1
                
        elif self.rebalance_frequency == 'annual':
            current_year = start.year
            while current_year <= end.year:
                jan_date = pd.Timestamp(year=current_year, month=1, day=1)
                if start <= jan_date <= end:
                    dates.append(jan_date)
                current_year += 1
                
        elif self.rebalance_frequency == 'quarterly':
            current_year = start.year
            while current_year <= end.year:
                for month in [1, 4, 7, 10]:
                    q_date = pd.Timestamp(year=current_year, month=month, day=1)
                    if start <= q_date <= end:
                        dates.append(q_date)
                current_year += 1
        
        # Always include start date as first rebalance
        if start not in dates:
            dates = [start] + dates
        
        return sorted(dates)
    
    def load_all_stock_data(
        self,
        data_dir: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ALL available stock data from CSV files.
        This caches the data for reuse in rolling selection.
        """
        if self.all_stock_data:
            return self.all_stock_data
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob('*.csv'))
        
        print(f"📂 Loading all stock data from {data_dir}...")
        loaded_count = 0
        
        for csv_file in csv_files:
            stock_name = csv_file.stem.replace('_5min', '').replace('_daily', '')
            
            try:
                df = pd.read_csv(csv_file)
                
                # Normalize column names
                df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date', 'datetime', 'time'] else c for c in df.columns]
                
                # Handle date column
                date_col = None
                for col in ['Date', 'Datetime', 'Time']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col is None:
                    continue
                
                df['Date'] = pd.to_datetime(df[date_col])
                
                # Remove timezone for consistent comparison
                if df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)
                
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                # Resample to daily if intraday
                if len(df) > 0:
                    daily_count = df['Date'].dt.date.nunique()
                    if len(df) / max(daily_count, 1) > 10:
                        df = df.set_index('Date').resample('D').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna().reset_index()
                
                if len(df) >= 50:  # Minimum data requirement
                    self.all_stock_data[stock_name] = df
                    loaded_count += 1
                    if loaded_count % 50 == 0:
                        print(f"   Loaded {loaded_count} stocks...")
                    
            except Exception as e:
                continue
        
        print(f"   ✓ Loaded {len(self.all_stock_data)} stocks total")
        return self.all_stock_data

    def select_stocks_for_period(
        self,
        selection_date: pd.Timestamp,
        target_count: int = 20
    ) -> Tuple[List[str], Dict]:
        """
        Select stocks for a specific period using only data available at selection_date.
        
        Uses:
        1. NIFTY 50 composition as of selection_date (historical)
        2. Lookback period ending at selection_date for metrics
        """
        self.selection_log = []
        self.stock_metrics = {}
        
        # Get NIFTY 50 composition for this date
        nifty_universe = self.get_nifty50_for_date(selection_date)
        self.selection_log.append(f"Date: {selection_date.strftime('%Y-%m-%d')}")
        self.selection_log.append(f"NIFTY 50 composition: {len(nifty_universe)} stocks (as of {selection_date.year})")
        
        # Filter to stocks we have data for
        available_in_universe = [s for s in nifty_universe if s in self.all_stock_data]
        self.selection_log.append(f"Available in data: {len(available_in_universe)} stocks")
        
        if len(available_in_universe) == 0:
            return [], {'error': 'No stocks available'}
        
        # Calculate metrics using only data up to selection_date
        self._calculate_stock_metrics_for_date(available_in_universe, selection_date)
        
        # Apply filters
        selected = self._apply_filters(available_in_universe, target_count)
        
        report = {
            'selection_date': selection_date,
            'nifty_composition': nifty_universe,
            'available_stocks': available_in_universe,
            'selected_stocks': selected,
            'selection_log': self.selection_log.copy(),
            'stock_metrics': {s: self.stock_metrics[s] for s in selected if s in self.stock_metrics}
        }
        
        return selected, report
    
    def _calculate_stock_metrics_for_date(
        self,
        stocks: List[str],
        selection_date: pd.Timestamp
    ):
        """Calculate metrics for stocks using only data up to selection_date."""
        
        for stock_name in stocks:
            if stock_name not in self.all_stock_data:
                continue
            
            df = self.all_stock_data[stock_name].copy()
            
            # Normalize dates - remove timezone if present for comparison
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            df['Date'] = df['Date'].dt.normalize()
            
            # Make selection_date timezone-naive too
            sel_date = pd.to_datetime(selection_date)
            if sel_date.tz is not None:
                sel_date = sel_date.tz_localize(None)
            sel_date = sel_date.normalize()
            
            # Only use data up to selection_date
            df = df[df['Date'] <= sel_date].copy()
            
            if len(df) < 50:
                continue
            
            # Use last lookback_days for calculations
            if len(df) > self.lookback_days:
                calc_df = df.tail(self.lookback_days).copy()
            else:
                calc_df = df.copy()
            
            # Calculate metrics
            calc_df['Returns'] = calc_df['Close'].pct_change()
            daily_returns = calc_df['Returns'].dropna()
            
            if len(daily_returns) < 20:
                continue
            
            total_return = (calc_df['Close'].iloc[-1] / calc_df['Close'].iloc[0] - 1) * 100
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            
            cummax = calc_df['Close'].cummax()
            drawdown = (calc_df['Close'] - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            self.stock_metrics[stock_name] = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'avg_volume': calc_df['Volume'].mean(),
                'data_days': len(df)
            }

    def load_and_filter_stocks_rolling(
        self,
        data_dir: str,
        start_date: str,
        end_date: str,
        target_count: int = 20
    ) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, List[str]], Dict]:
        """
        Load stocks with ROLLING bi-annual selection.
        
        Returns:
            - combined_data: All stock data for all periods
            - period_selections: Dict mapping each rebalance date to selected stocks
            - full_report: Complete selection history
        """
        print("=" * 80)
        print("🔄 ROLLING STOCK SELECTION (Bi-Annual)")
        print("=" * 80)
        
        # Load all data first
        self.load_all_stock_data(data_dir, start_date, end_date)
        
        # Get rebalancing dates
        rebalance_dates = self.get_rebalance_dates(start_date, end_date)
        print(f"\n📅 Rebalancing dates ({self.rebalance_frequency}):")
        for d in rebalance_dates:
            print(f"   • {d.strftime('%Y-%m-%d')}")
        
        # Select stocks for each period
        period_selections = {}
        all_reports = []
        all_selected_stocks = set()
        
        print(f"\n🔍 Selecting stocks for each period...")
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"\n--- Period {i+1}: {rebal_date.strftime('%Y-%m-%d')} ---")
            
            selected, report = self.select_stocks_for_period(rebal_date, target_count)
            period_selections[rebal_date] = selected
            all_reports.append(report)
            all_selected_stocks.update(selected)
            
            print(f"   Selected: {len(selected)} stocks")
            if selected:
                print(f"   Top 5: {selected[:5]}")
        
        # Track changes between periods
        print(f"\n📊 Selection Changes Over Time:")
        for i in range(1, len(rebalance_dates)):
            prev_date = rebalance_dates[i-1]
            curr_date = rebalance_dates[i]
            prev_stocks = set(period_selections[prev_date])
            curr_stocks = set(period_selections[curr_date])
            
            added = curr_stocks - prev_stocks
            removed = prev_stocks - curr_stocks
            
            print(f"\n   {prev_date.strftime('%Y-%m')} → {curr_date.strftime('%Y-%m')}:")
            print(f"   Added ({len(added)}): {list(added)[:5]}{'...' if len(added) > 5 else ''}")
            print(f"   Removed ({len(removed)}): {list(removed)[:5]}{'...' if len(removed) > 5 else ''}")
        
        # Combine data for ALL stocks that were ever selected
        print(f"\n📦 Combining data for {len(all_selected_stocks)} unique stocks...")
        combined_data = self._combine_stock_data(self.all_stock_data, list(all_selected_stocks))
        
        full_report = {
            'rebalance_frequency': self.rebalance_frequency,
            'rebalance_dates': rebalance_dates,
            'period_selections': period_selections,
            'all_selected_stocks': list(all_selected_stocks),
            'period_reports': all_reports,
            'total_unique_stocks': len(all_selected_stocks)
        }
        
        return combined_data, period_selections, full_report

    # Keep original method for backward compatibility
    def load_and_filter_stocks(
        self,
        data_dir: str,
        start_date: str,
        end_date: str,
        target_count: int = 20,
        universe: str = 'nifty50',
        selection_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Original method - single selection (backward compatible).
        For rolling selection, use load_and_filter_stocks_rolling() instead.
        """
        self.selection_log = []  # Reset log
        self.stock_metrics = {}  # Reset metrics
        
        data_path = Path(data_dir)
        
        # Step 1: Determine universe
        if universe == 'nifty50':
            target_stocks = self.NIFTY_50_STOCKS
            self.selection_log.append(f"Universe: NIFTY 50 ({len(target_stocks)} stocks)")
        else:
            target_stocks = None
            self.selection_log.append(f"Universe: All available stocks")
        
        # Step 2: Load all available data
        print(f"📂 Loading data from {data_dir}...")
        all_stock_data = {}
        available_stocks = []
        
        csv_files = list(data_path.glob('*.csv'))
        
        loaded_count = 0
        for csv_file in csv_files:
            stock_name = csv_file.stem.replace('_5min', '').replace('_daily', '')
            
            # If using NIFTY 50, only load those stocks
            if target_stocks is not None and stock_name not in target_stocks:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Normalize column names to handle different cases
                df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date', 'datetime', 'time'] else c for c in df.columns]
                
                # Handle different date column names
                date_col = None
                for col in ['Date', 'Datetime', 'Time']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col is None:
                    continue
                
                df['Date'] = pd.to_datetime(df[date_col])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                # Resample to daily if needed (intraday data has many rows per day)
                if len(df) > 0:
                    daily_count = df['Date'].dt.date.nunique()
                    if len(df) / max(daily_count, 1) > 10:  # More than 10 rows per day = intraday
                        df = df.set_index('Date').resample('D').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna().reset_index()
                
                if len(df) >= self.min_data_days:
                    all_stock_data[stock_name] = df
                    available_stocks.append(stock_name)
                    loaded_count += 1
                    if loaded_count % 10 == 0:
                        print(f"   Loaded {loaded_count} stocks...")
                    
            except Exception as e:
                continue
        
        self.selection_log.append(f"Stocks with sufficient data (≥{self.min_data_days} days): {len(available_stocks)}")
        print(f"   Found {len(available_stocks)} stocks with sufficient data")
        
        if len(available_stocks) == 0:
            print("⚠️ No stocks found! Check data directory and file format.")
            return pd.DataFrame(), [], {'error': 'No stocks found'}
        
        if len(available_stocks) < target_count:
            print(f"⚠️ Only {len(available_stocks)} stocks available, less than target {target_count}")
        
        # Step 3: Calculate metrics for each stock (using data up to selection_date only!)
        print(f"📊 Calculating selection metrics...")
        self._calculate_stock_metrics(all_stock_data, selection_date)
        
        # Step 4: Apply filters
        print(f"🔍 Applying filters...")
        selected_stocks = self._apply_filters(available_stocks, target_count)
        
        # Step 5: Combine data for selected stocks
        print(f"📦 Combining data for {len(selected_stocks)} selected stocks...")
        combined_data = self._combine_stock_data(all_stock_data, selected_stocks)
        
        # Step 6: Generate report
        report = self._generate_selection_report(available_stocks, selected_stocks)
        
        return combined_data, selected_stocks, report
    
    def _calculate_stock_metrics(
        self, 
        stock_data: Dict[str, pd.DataFrame],
        selection_date: Optional[str] = None
    ):
        """Calculate metrics for each stock using ONLY data up to selection_date."""
        
        for stock_name, df in stock_data.items():
            # CRITICAL: Only use data up to selection_date to avoid look-ahead bias
            if selection_date:
                df = df[df['Date'] <= selection_date].copy()
            
            if len(df) < 50:  # Need minimum data
                continue
            
            # Use last lookback_days for calculations
            if len(df) > self.lookback_days:
                calc_df = df.tail(self.lookback_days).copy()
            else:
                calc_df = df.copy()
            
            # Calculate returns
            calc_df['Returns'] = calc_df['Close'].pct_change()
            daily_returns = calc_df['Returns'].dropna()
            
            if len(daily_returns) < 20:
                continue
            
            # Calculate metrics
            total_return = (calc_df['Close'].iloc[-1] / calc_df['Close'].iloc[0] - 1) * 100
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            
            # Max drawdown
            cummax = calc_df['Close'].cummax()
            drawdown = (calc_df['Close'] - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            self.stock_metrics[stock_name] = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'avg_volume': calc_df['Volume'].mean(),
                'data_days': len(df)
            }
    
    def _apply_filters(
        self, 
        available_stocks: List[str], 
        target_count: int
    ) -> List[str]:
        """Apply filtering criteria to select stocks."""
        
        # Only consider stocks with calculated metrics
        candidates = [s for s in available_stocks if s in self.stock_metrics]
        
        if len(candidates) == 0:
            return available_stocks[:target_count]
        
        # Create DataFrame for easier filtering
        metrics_df = pd.DataFrame(self.stock_metrics).T
        metrics_df = metrics_df.loc[candidates]
        
        initial_count = len(metrics_df)
        self.selection_log.append(f"Candidates with metrics: {initial_count}")
        
        # Filter 1: Remove most volatile stocks (top 25%)
        vol_threshold = np.percentile(metrics_df['volatility'], self.max_volatility_percentile)
        low_vol_mask = metrics_df['volatility'] <= vol_threshold
        removed_vol = (~low_vol_mask).sum()
        self.selection_log.append(f"Volatility filter (≤{vol_threshold:.1f}%): removed {removed_vol} stocks")
        
        # Filter 2: Remove worst performing stocks (bottom 25%)
        ret_threshold = np.percentile(metrics_df['total_return'], self.min_return_percentile)
        good_ret_mask = metrics_df['total_return'] >= ret_threshold
        removed_ret = (~good_ret_mask).sum()
        self.selection_log.append(f"Return filter (≥{ret_threshold:.1f}%): removed {removed_ret} stocks")
        
        # Filter 3: Remove low volume stocks (bottom 25%)
        vol_threshold_liq = np.percentile(metrics_df['avg_volume'], self.min_volume_percentile)
        liquid_mask = metrics_df['avg_volume'] >= vol_threshold_liq
        removed_liq = (~liquid_mask).sum()
        self.selection_log.append(f"Liquidity filter: removed {removed_liq} stocks")
        
        # Combine filters
        combined_mask = low_vol_mask & good_ret_mask & liquid_mask
        filtered_df = metrics_df[combined_mask]
        
        self.selection_log.append(f"After all filters: {len(filtered_df)} stocks remain")
        
        # Final selection: Sort by Sharpe ratio and take top N
        if len(filtered_df) >= target_count:
            final_df = filtered_df.sort_values('sharpe', ascending=False).head(target_count)
            selected = final_df.index.tolist()
            self.selection_log.append(f"Selected top {target_count} by Sharpe ratio")
        elif len(filtered_df) > 0:
            selected = filtered_df.index.tolist()
            self.selection_log.append(f"Selected all {len(selected)} remaining stocks")
        else:
            # Fallback: just sort by Sharpe without filters
            selected = metrics_df.sort_values('sharpe', ascending=False).head(target_count).index.tolist()
            self.selection_log.append(f"Filters too strict - using top {len(selected)} by Sharpe")
        
        return selected
    
    def _combine_stock_data(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        selected_stocks: List[str]
    ) -> pd.DataFrame:
        """Combine data for selected stocks into single DataFrame."""
        
        combined_frames = []
        
        for stock_name in selected_stocks:
            if stock_name in stock_data:
                df = stock_data[stock_name].copy()
                df['Stock'] = stock_name
                combined_frames.append(df)
        
        if combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True)
            combined = combined.sort_values(['Date', 'Stock']).reset_index(drop=True)
            return combined
        
        return pd.DataFrame()
    
    def _generate_selection_report(
        self, 
        available_stocks: List[str], 
        selected_stocks: List[str]
    ) -> Dict:
        """Generate a detailed selection report."""
        
        report = {
            'total_available': len(available_stocks),
            'total_selected': len(selected_stocks),
            'selected_stocks': selected_stocks,
            'selection_log': self.selection_log,
            'stock_metrics': {s: self.stock_metrics[s] for s in selected_stocks if s in self.stock_metrics}
        }
        
        return report
    
    def print_selection_report(self, report: Dict):
        """Pretty print the selection report."""
        
        print("\n" + "=" * 80)
        print("📊 SMART STOCK SELECTION REPORT")
        print("=" * 80)
        
        print(f"\n📌 Selection Summary:")
        print(f"   Available: {report['total_available']} → Selected: {report['total_selected']}")
        
        print(f"\n🔍 Selection Process:")
        for log_entry in report['selection_log']:
            print(f"   • {log_entry}")
        
        if report['stock_metrics']:
            print(f"\n📈 Selected Stocks:")
            print("-" * 80)
            print(f"{'Stock':<15} {'Return%':>10} {'Vol%':>10} {'Sharpe':>10} {'MaxDD%':>10}")
            print("-" * 80)
            
            sorted_stocks = sorted(
                report['stock_metrics'].items(),
                key=lambda x: x[1].get('sharpe', 0),
                reverse=True
            )
            
            for stock, m in sorted_stocks:
                print(f"{stock:<15} {m['total_return']:>10.1f} {m['volatility']:>10.1f} {m['sharpe']:>10.2f} {m['max_drawdown']:>10.1f}")
            
            print("-" * 80)
            
            # Summary stats
            avg_ret = np.mean([m['total_return'] for m in report['stock_metrics'].values()])
            avg_vol = np.mean([m['volatility'] for m in report['stock_metrics'].values()])
            avg_sharpe = np.mean([m['sharpe'] for m in report['stock_metrics'].values()])
            
            print(f"\n   Portfolio Avg: Return={avg_ret:.1f}%, Vol={avg_vol:.1f}%, Sharpe={avg_sharpe:.2f}")
