"""
Feature engineering module for stock and market features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class FeatureEngineering:
    """
    Feature engineering for stock-level and market features.
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Initialize feature engineering.
        
        Args:
            lookback_window: Number of days for rolling calculations
        """
        self.lookback_window = lookback_window
    
    def compute_stock_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute stock-level features.
        
        Args:
            prices_df: DataFrame with columns [Date, Stock, Open, High, Low, Close, Volume]
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        for stock in prices_df['Stock'].unique():
            stock_data = prices_df[prices_df['Stock'] == stock].copy()
            stock_data = stock_data.sort_values('Date')
            
            # Price-based features
            stock_data['Returns'] = stock_data['Close'].pct_change()
            stock_data['LogReturns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            
            # Momentum features
            stock_data['Momentum_5'] = stock_data['Close'].pct_change(5)
            stock_data['Momentum_10'] = stock_data['Close'].pct_change(10)
            stock_data['Momentum_20'] = stock_data['Close'].pct_change(20)
            
            # Volatility features
            stock_data['Volatility_5'] = stock_data['Returns'].rolling(5).std()
            stock_data['Volatility_10'] = stock_data['Returns'].rolling(10).std()
            stock_data['Volatility_20'] = stock_data['Returns'].rolling(20).std()
            
            # Price range features
            stock_data['HighLow_Range'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
            stock_data['OpenClose_Range'] = (stock_data['Close'] - stock_data['Open']) / stock_data['Open']
            
            # Moving averages
            stock_data['SMA_5'] = stock_data['Close'].rolling(5).mean()
            stock_data['SMA_10'] = stock_data['Close'].rolling(10).mean()
            stock_data['SMA_20'] = stock_data['Close'].rolling(20).mean()
            stock_data['Price_to_SMA5'] = stock_data['Close'] / stock_data['SMA_5'] - 1
            stock_data['Price_to_SMA10'] = stock_data['Close'] / stock_data['SMA_10'] - 1
            stock_data['Price_to_SMA20'] = stock_data['Close'] / stock_data['SMA_20'] - 1
            
            # Volume features
            stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
            stock_data['Volume_SMA_5'] = stock_data['Volume'].rolling(5).mean()
            stock_data['Volume_Ratio'] = stock_data['Volume'] / stock_data['Volume_SMA_5']
            
            # RSI (Relative Strength Index)
            stock_data['RSI_14'] = self._compute_rsi(stock_data['Close'], 14)
            
            features_list.append(stock_data)
        
        return pd.concat(features_list, ignore_index=True)
    
    def compute_sector_relative_features(
        self, 
        features_df: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Compute sector-relative features.
        
        Args:
            features_df: DataFrame with stock features
            sector_mapping: Dictionary mapping stock symbols to sectors
            
        Returns:
            DataFrame with sector-relative features added
        """
        if sector_mapping is None:
            return features_df
        
        features_df = features_df.copy()
        features_df['Sector'] = features_df['Stock'].map(sector_mapping)
        
        # Compute sector averages
        for date in features_df['Date'].unique():
            date_mask = features_df['Date'] == date
            date_data = features_df[date_mask]
            
            for sector in date_data['Sector'].unique():
                if pd.isna(sector):
                    continue
                    
                sector_mask = date_mask & (features_df['Sector'] == sector)
                
                # Sector-relative return
                sector_return = date_data[date_data['Sector'] == sector]['Returns'].mean()
                features_df.loc[sector_mask, 'Sector_Relative_Return'] = (
                    features_df.loc[sector_mask, 'Returns'] - sector_return
                )
                
                # Sector-relative momentum
                sector_momentum = date_data[date_data['Sector'] == sector]['Momentum_10'].mean()
                features_df.loc[sector_mask, 'Sector_Relative_Momentum'] = (
                    features_df.loc[sector_mask, 'Momentum_10'] - sector_momentum
                )
        
        return features_df
    
    def compute_market_features(
        self,
        nifty_prices: pd.DataFrame,
        usdinr_prices: Optional[pd.DataFrame] = None,
        crude_prices: Optional[pd.DataFrame] = None,
        bond_yields: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute market-level context features.
        
        Args:
            nifty_prices: NIFTY index prices
            usdinr_prices: USD-INR exchange rate
            crude_prices: Crude oil prices
            bond_yields: Government bond yields
            
        Returns:
            DataFrame with market features
        """
        market_features = pd.DataFrame()
        nifty_prices = nifty_prices.sort_values('Date').reset_index(drop=True)
        market_features['Date'] = nifty_prices['Date']
        
        # NIFTY features
        market_features['NIFTY_Return'] = nifty_prices['Close'].pct_change()
        market_features['NIFTY_Volatility'] = market_features['NIFTY_Return'].rolling(20).std()
        market_features['NIFTY_Momentum_5'] = nifty_prices['Close'].pct_change(5)
        market_features['NIFTY_Momentum_10'] = nifty_prices['Close'].pct_change(10)
        
        # Track which features are available
        self._available_market_features = [
            'NIFTY_Return', 'NIFTY_Volatility',
            'NIFTY_Momentum_5', 'NIFTY_Momentum_10'
        ]
        
        # USD-INR features
        if usdinr_prices is not None and len(usdinr_prices) > 0:
            usdinr_prices = usdinr_prices.sort_values('Date').reset_index(drop=True)
            # Merge on date to align
            usdinr_aligned = pd.merge(
                market_features[['Date']], 
                usdinr_prices[['Date', 'Close']], 
                on='Date', 
                how='left'
            )
            usdinr_aligned['Close'] = usdinr_aligned['Close'].ffill()
            market_features['USDINR_Delta'] = usdinr_aligned['Close'].pct_change()
            market_features['USDINR_Volatility'] = market_features['USDINR_Delta'].rolling(20).std()
            self._available_market_features.extend(['USDINR_Delta', 'USDINR_Volatility'])
        
        # Crude features
        if crude_prices is not None and len(crude_prices) > 0:
            crude_prices = crude_prices.sort_values('Date').reset_index(drop=True)
            # Merge on date to align
            crude_aligned = pd.merge(
                market_features[['Date']], 
                crude_prices[['Date', 'Close']], 
                on='Date', 
                how='left'
            )
            crude_aligned['Close'] = crude_aligned['Close'].ffill()
            market_features['Crude_Delta'] = crude_aligned['Close'].pct_change()
            market_features['Crude_Volatility'] = market_features['Crude_Delta'].rolling(20).std()
            self._available_market_features.extend(['Crude_Delta', 'Crude_Volatility'])
        
        # Bond yield features
        if bond_yields is not None and len(bond_yields) > 0:
            bond_yields = bond_yields.sort_values('Date').reset_index(drop=True)
            # Merge on date to align
            bond_aligned = pd.merge(
                market_features[['Date']], 
                bond_yields[['Date', 'Close']], 
                on='Date', 
                how='left'
            )
            bond_aligned['Close'] = bond_aligned['Close'].ffill()
            market_features['Rates_Delta'] = bond_aligned['Close'].diff()
            market_features['Rates_Level'] = bond_aligned['Close']
            self._available_market_features.extend(['Rates_Delta', 'Rates_Level'])
        
        return market_features
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names.
        
        Returns:
            List of feature names
        """
        return [
            'Returns', 'LogReturns',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'HighLow_Range', 'OpenClose_Range',
            'Price_to_SMA5', 'Price_to_SMA10', 'Price_to_SMA20',
            'Volume_Change', 'Volume_Ratio',
            'RSI_14'
        ]
    
    def get_market_feature_columns(self) -> List[str]:
        """
        Get list of market feature column names.
        Returns only the features that were computed (available).
        
        Returns:
            List of market feature names
        """
        # Return available features if compute_market_features was called
        if hasattr(self, '_available_market_features'):
            return self._available_market_features
        
        # Default full list (for backward compatibility)
        return [
            'NIFTY_Return', 'NIFTY_Volatility',
            'NIFTY_Momentum_5', 'NIFTY_Momentum_10',
            'USDINR_Delta', 'USDINR_Volatility',
            'Crude_Delta', 'Crude_Volatility',
            'Rates_Delta', 'Rates_Level'
        ]
