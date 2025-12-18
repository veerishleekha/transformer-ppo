"""
Market data loading and management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import warnings
import os
import glob

warnings.filterwarnings('ignore')


class MarketDataLoader:
    """
    Load and manage market data for NIFTY 50/100 stocks.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize market data loader.
        
        Args:
            data_dir: Optional path to local data directory (e.g., nifty_500_5min folder)
        """
        self.data_dir = data_dir
        self.nifty50_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
            'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
            'TITAN', 'BAJFINANCE', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO',
            'HCLTECH', 'TECHM', 'POWERGRID', 'NTPC', 'ONGC',
            'TATASTEEL', 'ADANIPORTS', 'JSWSTEEL', 'INDUSINDBK', 'TATAMOTORS',
            'BAJAJFINSV', 'COALINDIA', 'CIPLA', 'HINDALCO', 'GRASIM',
            'DRREDDY', 'EICHERMOT', 'BRITANNIA', 'HEROMOTOCO', 'DIVISLAB',
            'APOLLOHOSP', 'BPCL', 'ADANIENT', 'TATACONSUM', 'UPL',
            'BAJAJ-AUTO', 'SHRIRAMFIN', 'SBILIFE', 'LTIM', 'HDFCLIFE'
        ]
    
    def load_from_local_csv(
        self,
        data_dir: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample_to_daily: bool = True,
        max_stocks: int = 20
    ) -> pd.DataFrame:
        """
        Load stock data from local CSV files (5-minute data).
        
        Args:
            data_dir: Path to directory containing CSV files
            symbols: List of stock symbols to load (without .csv extension)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resample_to_daily: If True, resample 5-min data to daily OHLCV
            max_stocks: Maximum number of stocks to load
            
        Returns:
            DataFrame with columns [Date, Stock, Open, High, Low, Close, Volume]
        """
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Get available CSV files
        available_files = list(data_dir.glob("*.csv"))
        available_symbols = [f.stem for f in available_files]
        
        print(f"Found {len(available_symbols)} stocks in {data_dir}")
        
        # Filter to requested symbols or use defaults
        if symbols is None:
            # Use NIFTY 50 symbols that are available
            symbols = [s for s in self.nifty50_symbols if s in available_symbols]
            if len(symbols) < max_stocks:
                # Add more stocks if needed
                additional = [s for s in available_symbols if s not in symbols][:max_stocks - len(symbols)]
                symbols.extend(additional)
        
        symbols = symbols[:max_stocks]
        print(f"Loading {len(symbols)} stocks...")
        
        data_list = []
        loaded_count = 0
        
        for i, symbol in enumerate(symbols):
            csv_path = data_dir / f"{symbol}.csv"
            
            if not csv_path.exists():
                print(f"  Warning: {symbol}.csv not found, skipping...")
                continue
            
            try:
                # Read CSV
                df = pd.read_csv(csv_path)
                
                # Standardize column names
                df.columns = df.columns.str.lower()
                
                # Parse date column
                df['date'] = pd.to_datetime(df['date'])
                
                # Remove timezone if present
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                # Filter by date range
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                if len(df) == 0:
                    continue
                
                if resample_to_daily:
                    # Resample 5-minute data to daily OHLCV
                    df = df.set_index('date')
                    daily_df = df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    daily_df = daily_df.reset_index()
                    df = daily_df
                
                # Normalize date to remove time component
                df['date'] = pd.to_datetime(df['date']).dt.normalize()
                
                # Add stock symbol
                df['Stock'] = symbol
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Select required columns
                df = df[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                data_list.append(df)
                loaded_count += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  Loaded {i + 1}/{len(symbols)} stocks...")
                    
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
                continue
        
        if not data_list:
            raise ValueError("No data loaded from CSV files")
        
        result = pd.concat(data_list, ignore_index=True)
        result = result.sort_values(['Date', 'Stock']).reset_index(drop=True)
        
        print(f"\n✓ Successfully loaded {loaded_count} stocks")
        print(f"  Date range: {result['Date'].min().date()} to {result['Date'].max().date()}")
        print(f"  Total observations: {len(result):,}")
        
        return result
    
    def load_index_from_local(
        self,
        data_dir: str,
        index_symbol: str = 'NIFTY',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample_to_daily: bool = True
    ) -> pd.DataFrame:
        """
        Load index data from local CSV or create a synthetic index from loaded stocks.
        
        Args:
            data_dir: Path to directory containing CSV files
            index_symbol: Symbol of the index file (if available)
            start_date: Start date
            end_date: End date
            resample_to_daily: Resample to daily data
            
        Returns:
            DataFrame with index data
        """
        data_dir = Path(data_dir)
        index_path = data_dir / f"{index_symbol}.csv"
        
        # Try loading NIFTY index file if it exists
        if index_path.exists():
            try:
                df = pd.read_csv(index_path)
                df.columns = df.columns.str.lower()
                df['date'] = pd.to_datetime(df['date'])
                
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                if resample_to_daily:
                    df = df.set_index('date')
                    df = df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna().reset_index()
                
                df['date'] = pd.to_datetime(df['date']).dt.normalize()
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                print(f"✓ Loaded {index_symbol} index from CSV")
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
            except Exception as e:
                print(f"Could not load {index_symbol} index: {e}")
        
        # Create synthetic index from major stocks
        print("Creating synthetic NIFTY index from loaded stocks...")
        return self._create_synthetic_index(data_dir, start_date, end_date, resample_to_daily)
    
    def _create_synthetic_index(
        self,
        data_dir: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample_to_daily: bool = True
    ) -> pd.DataFrame:
        """
        Create a synthetic index by averaging major stocks.
        """
        # Load a few major stocks to create synthetic index
        major_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
                        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']
        
        data_dir = Path(data_dir)
        prices_list = []
        
        for symbol in major_stocks:
            csv_path = data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.lower()
                df['date'] = pd.to_datetime(df['date'])
                
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                if resample_to_daily:
                    df = df.set_index('date')
                    df = df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    df = df.reset_index()
                
                df['date'] = pd.to_datetime(df['date']).dt.normalize()
                df = df.set_index('date')
                prices_list.append(df['close'])
                
            except Exception:
                continue
        
        if not prices_list:
            raise ValueError("Could not create synthetic index - no stock data loaded")
        
        # Combine and normalize
        prices_df = pd.concat(prices_list, axis=1)
        prices_df.columns = range(len(prices_list))
        
        # Create equal-weighted index (normalized to start at 15000 like NIFTY)
        returns = prices_df.pct_change()
        avg_returns = returns.mean(axis=1)
        index_values = 15000 * (1 + avg_returns).cumprod()
        index_values.iloc[0] = 15000
        
        # Create OHLCV from daily returns
        result = pd.DataFrame({
            'Date': index_values.index,
            'Close': index_values.values,
        })
        
        # Approximate OHLV from Close
        result['Open'] = result['Close'].shift(1).fillna(result['Close'])
        result['High'] = result[['Open', 'Close']].max(axis=1) * 1.005
        result['Low'] = result[['Open', 'Close']].min(axis=1) * 0.995
        result['Volume'] = 1e9  # Placeholder
        
        result = result[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✓ Created synthetic index from {len(prices_list)} stocks")
        return result

    def load_stock_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample_data: bool = True
    ) -> pd.DataFrame:
        """
        Load stock price data.
        
        Args:
            symbols: List of stock symbols (default: NIFTY 50)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample_data: If True, generate synthetic data for demonstration
            
        Returns:
            DataFrame with columns [Date, Stock, Open, High, Low, Close, Volume]
        """
        if symbols is None:
            symbols = self.nifty50_symbols[:20]  # Use subset for demo
        
        if self.data_dir:
            # Try loading from local CSV files first
            try:
                return self.load_from_local_csv(self.data_dir, symbols, start_date, end_date)
            except Exception as e:
                print(f"Error loading from local CSV: {e}")
                print("Falling back to sample data...")
        
        if use_sample_data:
            return self._generate_sample_data(symbols, start_date, end_date)
        else:
            # Real data loading using yfinance
            try:
                import yfinance as yf
            except ImportError:
                print("yfinance not available, using sample data")
                return self._generate_sample_data(symbols, start_date, end_date)
            
            data_list = []
            
            print(f"Loading data for {len(symbols)} stocks using batch download...")
            try:
                # Use batch download which is more reliable
                df = yf.download(
                    symbols, 
                    start=start_date, 
                    end=end_date,
                    progress=True,
                    threads=True,
                    group_by='ticker'
                )
                
                if df.empty:
                    print("WARNING: yfinance returned empty data, falling back to sample data")
                    return self._generate_sample_data(symbols, start_date, end_date)
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        if len(symbols) == 1:
                            stock_df = df.copy()
                        else:
                            if symbol not in df.columns.get_level_values(0):
                                continue
                            stock_df = df[symbol].copy()
                        
                        if stock_df.empty or stock_df['Close'].isna().all():
                            continue
                        
                        stock_df = stock_df.reset_index()
                        stock_df['Stock'] = symbol.replace('.NS', '')
                        
                        # Handle timezone-aware datetime
                        if hasattr(stock_df['Date'].dt, 'tz') and stock_df['Date'].dt.tz is not None:
                            stock_df['Date'] = stock_df['Date'].dt.tz_localize(None)
                        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.normalize()
                        
                        # Drop rows with NaN Close prices
                        stock_df = stock_df.dropna(subset=['Close'])
                        
                        if len(stock_df) > 0:
                            data_list.append(stock_df[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']])
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Batch download failed: {e}")
                print("Falling back to individual downloads...")
                
                # Fallback to individual downloads
                for i, symbol in enumerate(symbols):
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=start_date, end=end_date)
                        
                        if len(hist) > 0:
                            hist = hist.reset_index()
                            hist['Stock'] = symbol.replace('.NS', '')
                            if hist['Date'].dt.tz is not None:
                                hist['Date'] = hist['Date'].dt.tz_localize(None)
                            hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
                            data_list.append(hist[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']])
                        
                        if (i + 1) % 5 == 0:
                            print(f"  Loaded {i + 1}/{len(symbols)} stocks...")
                    except Exception as e:
                        continue
            
            if data_list:
                result = pd.concat(data_list, ignore_index=True)
                print(f"✓ Successfully loaded {result['Stock'].nunique()} stocks with {len(result)} observations")
                return result
            else:
                print("WARNING: No data loaded from yfinance, falling back to sample data")
                return self._generate_sample_data(symbols, start_date, end_date)
    
    def _generate_sample_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic stock data for demonstration.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with synthetic price data
        """
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2023-12-31'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)
        
        data_list = []
        np.random.seed(42)
        
        for i, symbol in enumerate(symbols):
            # Generate realistic stock prices using geometric brownian motion
            initial_price = 100 + np.random.uniform(50, 450)
            mu = 0.0002  # drift
            sigma = 0.02  # volatility
            
            returns = np.random.normal(mu, sigma, len(dates))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # Add some trend
            trend = np.linspace(0, 0.3 * np.random.randn(), len(dates))
            prices = prices * np.exp(trend)
            
            # Generate OHLC
            for j, (date, close) in enumerate(zip(dates, prices)):
                daily_vol = sigma * np.sqrt(1/252)
                high = close * (1 + abs(np.random.normal(0, daily_vol)))
                low = close * (1 - abs(np.random.normal(0, daily_vol)))
                open_price = close * (1 + np.random.normal(0, daily_vol/2))
                volume = np.random.uniform(1e6, 1e7)
                
                data_list.append({
                    'Date': date,
                    'Stock': symbol.replace('.NS', '') if '.NS' in symbol else f'STOCK{i}',
                    'Open': open_price,
                    'High': max(open_price, close, high),
                    'Low': min(open_price, close, low),
                    'Close': close,
                    'Volume': volume
                })
        
        return pd.DataFrame(data_list)
    
    def _generate_sample_index_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic NIFTY index data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with synthetic index data
        """
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2023-12-31'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)
        
        np.random.seed(42)
        initial_price = 15000
        mu = 0.0003
        sigma = 0.015
        
        returns = np.random.normal(mu, sigma, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))
        
        data_list = []
        for date, close in zip(dates, prices):
            daily_vol = sigma * np.sqrt(1/252)
            high = close * (1 + abs(np.random.normal(0, daily_vol)))
            low = close * (1 - abs(np.random.normal(0, daily_vol)))
            open_price = close * (1 + np.random.normal(0, daily_vol/2))
            volume = np.random.uniform(1e8, 5e8)
            
            data_list.append({
                'Date': date,
                'Open': open_price,
                'High': max(open_price, close, high),
                'Low': min(open_price, close, low),
                'Close': close,
                'Volume': volume
            })
        
        return pd.DataFrame(data_list)
    
    def load_nifty_index(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample_data: bool = True
    ) -> pd.DataFrame:
        """
        Load NIFTY 50 index data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample_data: If True, generate synthetic data
            
        Returns:
            DataFrame with NIFTY index prices
        """
        if use_sample_data:
            return self._generate_sample_index_data(start_date, end_date)
        else:
            try:
                import yfinance as yf
                print("Loading NIFTY 50 index data...")
                
                # Try batch download first (more reliable)
                df = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    # Fallback to Ticker method
                    nifty = yf.Ticker("^NSEI")
                    df = nifty.history(start=start_date, end=end_date)
                
                if df.empty or len(df) == 0:
                    print("Warning: Could not load NIFTY index, using sample data")
                    return self._generate_sample_index_data(start_date, end_date)
                
                df = df.reset_index()
                
                # Handle different column name formats
                date_col = 'Date' if 'Date' in df.columns else 'index'
                if date_col == 'index':
                    df = df.rename(columns={'index': 'Date'})
                
                # Handle timezone-aware datetime
                if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                
                print(f"✓ Loaded NIFTY index: {len(df)} days")
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                print(f"Warning: Could not load NIFTY index ({e}), using sample data")
                return self._generate_sample_index_data(start_date, end_date)
    
    def load_usdinr(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load USD/INR exchange rate data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample_data: If True, generate synthetic data
            
        Returns:
            DataFrame with USD/INR rates or None if unavailable
        """
        if use_sample_data:
            return self._generate_sample_forex_data(start_date, end_date)
        else:
            try:
                import yfinance as yf
                print("Loading USD/INR data...")
                usdinr = yf.Ticker("USDINR=X")
                hist = usdinr.history(start=start_date, end=end_date)
                if len(hist) > 0:
                    hist = hist.reset_index()
                    if hist['Date'].dt.tz is not None:
                        hist['Date'] = hist['Date'].dt.tz_localize(None)
                    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
                    print(f"✓ Loaded USD/INR: {len(hist)} days")
                    return hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                return None
            except Exception as e:
                print(f"Warning: Could not load USD/INR: {e}")
                return None
    
    def load_crude_oil(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load crude oil price data (Brent).
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample_data: If True, generate synthetic data
            
        Returns:
            DataFrame with crude oil prices or None if unavailable
        """
        if use_sample_data:
            return self._generate_sample_crude_data(start_date, end_date)
        else:
            try:
                import yfinance as yf
                print("Loading Crude Oil (Brent) data...")
                crude = yf.Ticker("BZ=F")  # Brent Crude
                hist = crude.history(start=start_date, end=end_date)
                if len(hist) > 0:
                    hist = hist.reset_index()
                    if hist['Date'].dt.tz is not None:
                        hist['Date'] = hist['Date'].dt.tz_localize(None)
                    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
                    print(f"✓ Loaded Crude Oil: {len(hist)} days")
                    return hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                return None
            except Exception as e:
                print(f"Warning: Could not load Crude Oil: {e}")
                return None
    
    def load_india_10y_bond(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sample_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load India 10-year government bond yield data.
        Note: Direct bond yield data may be limited, using proxy.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_sample_data: If True, generate synthetic data
            
        Returns:
            DataFrame with bond yields or None if unavailable
        """
        if use_sample_data:
            return self._generate_sample_bond_data(start_date, end_date)
        else:
            # India 10Y bond data is not directly available on yfinance
            # Using synthetic approximation based on typical behavior
            print("Note: India 10Y bond data using synthetic approximation")
            return self._generate_sample_bond_data(start_date, end_date)
    
    def _generate_sample_forex_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic USD/INR data."""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2023-12-31'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)
        
        np.random.seed(43)
        initial_rate = 74.0
        mu = 0.0001
        sigma = 0.005
        
        returns = np.random.normal(mu, sigma, len(dates))
        rates = initial_rate * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': rates * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': rates * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'Low': rates * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'Close': rates,
            'Volume': np.zeros(len(dates))
        })
        return data
    
    def _generate_sample_crude_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic crude oil data."""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2023-12-31'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)
        
        np.random.seed(44)
        initial_price = 65.0
        mu = 0.0002
        sigma = 0.025
        
        returns = np.random.normal(mu, sigma, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.uniform(1e5, 5e5, len(dates))
        })
        return data
    
    def _generate_sample_bond_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic bond yield data."""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2023-12-31'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.bdate_range(start=start, end=end)
        
        np.random.seed(45)
        initial_yield = 6.5
        mu = 0.0
        sigma = 0.01
        
        changes = np.random.normal(mu, sigma, len(dates))
        yields = initial_yield + np.cumsum(changes)
        yields = np.clip(yields, 4.0, 9.0)  # Reasonable yield range
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': yields + np.random.normal(0, 0.02, len(dates)),
            'High': yields + np.abs(np.random.normal(0, 0.03, len(dates))),
            'Low': yields - np.abs(np.random.normal(0, 0.03, len(dates))),
            'Close': yields,
            'Volume': np.zeros(len(dates))
        })
        return data
    
    def get_sector_mapping(self) -> dict:
        """
        Get mapping of stocks to sectors.
        
        Returns:
            Dictionary mapping stock symbols to sectors
        """
        # Simplified sector mapping for demonstration
        sector_map = {
            'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Finance', 
            'INFY': 'IT', 'ICICIBANK': 'Finance', 'HINDUNILVR': 'FMCG',
            'ITC': 'FMCG', 'SBIN': 'Finance', 'BHARTIARTL': 'Telecom',
            'KOTAKBANK': 'Finance', 'LT': 'Infrastructure', 'AXISBANK': 'Finance',
            'ASIANPAINT': 'Materials', 'MARUTI': 'Auto', 'SUNPHARMA': 'Pharma',
            'TITAN': 'Consumer', 'BAJFINANCE': 'Finance', 'ULTRACEMCO': 'Materials',
            'NESTLEIND': 'FMCG', 'WIPRO': 'IT'
        }
        return sector_map
