"""
Data preprocessing and normalization utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Preprocess and normalize data for model training.
    """
    
    def __init__(self, clip_range: float = 5.0):
        """
        Initialize preprocessor.
        
        Args:
            clip_range: Range for clipping normalized values (default: 5.0 std deviations)
        """
        self.stock_scaler = StandardScaler()
        self.market_scaler = StandardScaler()
        self.feature_columns = []
        self.market_feature_columns = []
        self.fitted = False
        self.clip_range = clip_range
    
    def fit(
        self,
        stock_features: pd.DataFrame,
        market_features: pd.DataFrame,
        feature_columns: List[str],
        market_feature_columns: List[str]
    ):
        """
        Fit scalers on training data.
        
        Args:
            stock_features: Stock-level features DataFrame
            market_features: Market-level features DataFrame
            feature_columns: List of stock feature column names
            market_feature_columns: List of market feature column names
        """
        self.feature_columns = feature_columns
        self.market_feature_columns = market_feature_columns
        
        # Fit stock features scaler
        stock_data = stock_features[feature_columns].values
        stock_data = stock_data[~np.isnan(stock_data).any(axis=1)]
        if len(stock_data) > 0:
            self.stock_scaler.fit(stock_data)
        
        # Fit market features scaler
        market_data = market_features[market_feature_columns].values
        market_data = market_data[~np.isnan(market_data).any(axis=1)]
        if len(market_data) > 0:
            self.market_scaler.fit(market_data)
        
        self.fitted = True
    
    def transform_stock_features(
        self,
        stock_features: pd.DataFrame,
        fill_na: bool = True
    ) -> np.ndarray:
        """
        Transform stock features.
        
        Args:
            stock_features: Stock features DataFrame
            fill_na: Whether to fill NaN values
            
        Returns:
            Normalized stock features array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = stock_features[self.feature_columns].copy()
        
        if fill_na:
            data = data.fillna(0)
        
        # Transform
        transformed = self.stock_scaler.transform(data.values)
        
        # Clip extreme values to prevent outliers from affecting training
        # Clips to +/- clip_range standard deviations
        transformed = np.clip(transformed, -self.clip_range, self.clip_range)
        
        return transformed
    
    def transform_market_features(
        self,
        market_features: pd.DataFrame,
        fill_na: bool = True
    ) -> np.ndarray:
        """
        Transform market features.
        
        Args:
            market_features: Market features DataFrame
            fill_na: Whether to fill NaN values
            
        Returns:
            Normalized market features array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = market_features[self.market_feature_columns].copy()
        
        if fill_na:
            data = data.fillna(0)
        
        # Transform
        transformed = self.market_scaler.transform(data.values)
        
        # Clip extreme values to prevent outliers from affecting training
        # Clips to +/- clip_range standard deviations
        transformed = np.clip(transformed, -self.clip_range, self.clip_range)
        
        return transformed
    
    def create_sequences(
        self,
        stock_features: pd.DataFrame,
        market_features: pd.DataFrame,
        dates: pd.DatetimeIndex,
        stocks: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Create time-aligned sequences for training.
        
        Args:
            stock_features: Stock features DataFrame
            market_features: Market features DataFrame
            dates: Available dates
            stocks: List of stock symbols
            
        Returns:
            Tuple of (stock_sequences, market_sequences, valid_dates)
        """
        stock_sequences = []
        market_sequences = []
        valid_dates = []
        
        for date in dates:
            # Get stock features for this date
            date_stocks = stock_features[stock_features['Date'] == date]
            
            # Ensure we have all stocks
            if len(date_stocks) != len(stocks):
                continue
            
            # Sort by stock to ensure consistent ordering
            date_stocks = date_stocks.set_index('Stock').loc[stocks].reset_index()
            
            # Get market features for this date
            date_market = market_features[market_features['Date'] == date]
            
            if len(date_market) == 0:
                continue
            
            # Transform features
            try:
                stock_feat = self.transform_stock_features(date_stocks)
                market_feat = self.transform_market_features(date_market)
                
                # Check for NaN
                if np.isnan(stock_feat).any() or np.isnan(market_feat).any():
                    continue
                
                stock_sequences.append(stock_feat)
                market_sequences.append(market_feat[0])  # Single market vector per date
                valid_dates.append(date)
                
            except Exception as e:
                continue
        
        if len(stock_sequences) == 0:
            raise ValueError("No valid sequences created")
        
        return (
            np.array(stock_sequences),
            np.array(market_sequences),
            pd.DatetimeIndex(valid_dates)
        )
    
    def align_returns(
        self,
        stock_features: pd.DataFrame,
        dates: pd.DatetimeIndex,
        stocks: List[str],
        shift_forward: bool = True  # NEW PARAMETER: shift returns forward to avoid look-ahead bias
    ) -> np.ndarray:
        """
        Extract aligned returns for each date.
        
        IMPORTANT: To avoid look-ahead bias, returns should be shifted forward by 1 day.
        This means at time t, we get returns that will happen from t to t+1 (next-day returns).
        The model sees features at time t and predicts returns for the NEXT day.
        
        Args:
            stock_features: Stock features DataFrame with 'Returns' column
            dates: Dates to extract returns for
            stocks: List of stock symbols
            shift_forward: If True, returns[t] = return from t to t+1 (next-day return)
                          If False, returns[t] = return from t-1 to t (same-day return, HAS LOOK-AHEAD BIAS)
            
        Returns:
            Array of returns [T x N] where T=time, N=stocks
        """
        returns_list = []
        
        # Create a date-indexed lookup for faster access
        date_lookup = {}
        for date in stock_features['Date'].unique():
            date_stocks = stock_features[stock_features['Date'] == date]
            if len(date_stocks) == len(stocks):
                date_stocks = date_stocks.set_index('Stock').loc[stocks].reset_index()
                date_lookup[date] = date_stocks['Returns'].fillna(0).values
        
        # Convert dates to a list for indexing
        dates_list = list(dates)
        
        for i, date in enumerate(dates_list):
            if shift_forward:
                # Look ahead to the next date's return
                # At time t, we want the return that happens from t to t+1
                # This is stored in the features of t+1 (since Returns at t+1 = price[t+1]/price[t] - 1)
                if i + 1 < len(dates_list):
                    next_date = dates_list[i + 1]
                    if next_date in date_lookup:
                        returns_list.append(date_lookup[next_date])
                    else:
                        returns_list.append(np.zeros(len(stocks)))
                else:
                    # Last date - no next-day return available
                    returns_list.append(np.zeros(len(stocks)))
            else:
                # Same-day return (HAS LOOK-AHEAD BIAS - use only for debugging)
                if date in date_lookup:
                    returns_list.append(date_lookup[date])
                else:
                    returns_list.append(np.zeros(len(stocks)))
        
        return np.array(returns_list)
    
    def split_train_test(
        self,
        stock_sequences: np.ndarray,
        market_sequences: np.ndarray,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        train_ratio: float = 0.7
    ) -> Tuple:
        """
        Split data into train and test sets chronologically.
        
        Args:
            stock_sequences: Stock feature sequences
            market_sequences: Market feature sequences
            returns: Returns array
            dates: Date index
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_stock, train_market, train_returns, train_dates,
                     test_stock, test_market, test_returns, test_dates)
        """
        split_idx = int(len(dates) * train_ratio)
        
        return (
            stock_sequences[:split_idx],
            market_sequences[:split_idx],
            returns[:split_idx],
            dates[:split_idx],
            stock_sequences[split_idx:],
            market_sequences[split_idx:],
            returns[split_idx:],
            dates[split_idx:]
        )
