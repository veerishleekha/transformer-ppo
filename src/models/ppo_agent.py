"""
Combined PPO agent integrating all model components.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

from .transformer import StockFeatureEncoder, MarketContextEncoder
from .policy import PolicyNetwork
from .critic import ValueNetwork


class PPOAgent(nn.Module):
    """
    Complete PPO agent for portfolio optimization.
    
    Combines transformer encoder, policy network, and value network.
    
    AGGRESSIVE MODE: Set aggressive=True and reduce temperature for
    more concentrated, active portfolio weights.
    """
    
    def __init__(
        self,
        num_stock_features: int,
        num_market_features: int,
        num_stocks: int,
        stock_embedding_dim: int = 64,
        market_embedding_dim: int = 32,
        num_transformer_heads: int = 4,
        num_transformer_layers: int = 2,
        policy_hidden_dim: int = 64,
        value_hidden_dim: int = 128,
        dropout: float = 0.1,
        max_weight: float = 0.2,
        device: str = 'cpu',
        aggressive: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize PPO agent.
        
        Args:
            num_stock_features: Number of features per stock
            num_market_features: Number of market features
            num_stocks: Number of stocks in universe
            stock_embedding_dim: Dimension of stock embeddings
            market_embedding_dim: Dimension of market embeddings
            num_transformer_heads: Number of attention heads in transformer
            num_transformer_layers: Number of transformer layers
            policy_hidden_dim: Hidden dimension for policy network
            value_hidden_dim: Hidden dimension for value network
            dropout: Dropout probability
            max_weight: Maximum weight for any stock
            device: Device to run on ('cpu' or 'cuda')
            aggressive: If True, generate more concentrated active bets
            temperature: Temperature for weight concentration (lower = more concentrated)
        """
        super().__init__()
        
        self.num_stocks = num_stocks
        self.device = device
        self.aggressive = aggressive
        
        # Stock feature encoder (Transformer)
        self.stock_encoder = StockFeatureEncoder(
            num_features=num_stock_features,
            embedding_dim=stock_embedding_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Market context encoder
        self.market_encoder = MarketContextEncoder(
            num_market_features=num_market_features,
            embedding_dim=market_embedding_dim,
            dropout=dropout
        )
        
        # Policy network with aggressive mode support
        self.policy = PolicyNetwork(
            stock_embedding_dim=stock_embedding_dim,
            market_embedding_dim=market_embedding_dim,
            hidden_dim=policy_hidden_dim,
            dropout=dropout,
            max_weight=max_weight,
            aggressive=aggressive,
            temperature=temperature
        )
        
        # Value network
        self.value_net = ValueNetwork(
            stock_embedding_dim=stock_embedding_dim,
            market_embedding_dim=market_embedding_dim,
            hidden_dim=value_hidden_dim,
            dropout=dropout
        )
        
        self.to(device)
    
    def forward(
        self,
        stock_features: torch.Tensor,
        market_features: torch.Tensor,
        previous_weights: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete agent.
        
        Args:
            stock_features: Stock features [batch_size, num_stocks, num_features]
            market_features: Market features [batch_size, num_market_features]
            previous_weights: Previous portfolio weights [batch_size, num_stocks]
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (weights, log_probs, values, entropy)
        """
        # Encode stock features
        stock_embeddings = self.stock_encoder(stock_features)
        
        # Encode market context
        market_embedding = self.market_encoder(market_features)
        
        # Generate portfolio weights
        weights, log_probs, entropy = self.policy.get_action(
            stock_embeddings,
            market_embedding,
            previous_weights,
            deterministic
        )
        
        # Compute state value
        values = self.value_net(stock_embeddings, market_embedding)
        
        return weights, log_probs, values, entropy
    
    def get_action(
        self,
        stock_features: np.ndarray,
        market_features: np.ndarray,
        previous_weights: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Get action from current state (numpy interface).
        
        Args:
            stock_features: Stock features [num_stocks, num_features]
            market_features: Market features [num_market_features]
            previous_weights: Previous weights [num_stocks]
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (weights, log_prob)
        """
        # Convert to tensors
        stock_feat = torch.FloatTensor(stock_features).unsqueeze(0).to(self.device)
        market_feat = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)
        prev_weights = torch.FloatTensor(previous_weights).unsqueeze(0).to(self.device)
        
        # Get action
        with torch.no_grad():
            weights, log_prob, _, _ = self.forward(
                stock_feat,
                market_feat,
                prev_weights,
                deterministic
            )
        
        return weights[0].cpu().numpy(), log_prob[0].item()
    
    def evaluate_actions(
        self,
        stock_features: torch.Tensor,
        market_features: torch.Tensor,
        previous_weights: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            stock_features: Stock features
            market_features: Market features
            previous_weights: Previous weights
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        # Encode features
        stock_embeddings = self.stock_encoder(stock_features)
        market_embedding = self.market_encoder(market_features)
        
        # Evaluate actions
        log_probs, entropy, _ = self.policy.evaluate_actions(
            stock_embeddings,
            market_embedding,
            previous_weights,
            actions
        )
        
        # Compute values
        values = self.value_net(stock_embeddings, market_embedding)
        
        return log_probs, values, entropy
    
    def get_value(
        self,
        stock_features: torch.Tensor,
        market_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get state value.
        
        Args:
            stock_features: Stock features
            market_features: Market features
            
        Returns:
            State values
        """
        stock_embeddings = self.stock_encoder(stock_features)
        market_embedding = self.market_encoder(market_features)
        values = self.value_net(stock_embeddings, market_embedding)
        return values
