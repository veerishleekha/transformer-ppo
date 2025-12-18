"""
Value network (critic) for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    Value network that estimates V(s) for a given state.
    
    Uses attention pooling to aggregate stock embeddings and combines
    with market context to predict state value.
    """
    
    def __init__(
        self,
        stock_embedding_dim: int,
        market_embedding_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        """
        Initialize value network.
        
        Args:
            stock_embedding_dim: Dimension of stock embeddings
            market_embedding_dim: Dimension of market embeddings
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            use_attention_pooling: Whether to use attention pooling (vs mean pooling)
        """
        super().__init__()
        
        self.stock_embedding_dim = stock_embedding_dim
        self.market_embedding_dim = market_embedding_dim
        self.use_attention_pooling = use_attention_pooling
        
        # Attention pooling for stock embeddings
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(stock_embedding_dim)
        
        # Value network MLP
        # Input: pooled stock embedding + market embedding
        input_dim = stock_embedding_dim + market_embedding_dim
        
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        stock_embeddings: torch.Tensor,
        market_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute state value.
        
        Args:
            stock_embeddings: Stock embeddings [batch_size, num_stocks, stock_emb_dim]
            market_embedding: Market embedding [batch_size, market_emb_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        # Pool stock embeddings
        if self.use_attention_pooling:
            pooled_stocks = self.attention_pooling(stock_embeddings)
        else:
            # Simple mean pooling
            pooled_stocks = stock_embeddings.mean(dim=1)
        
        # Concatenate with market embedding
        combined = torch.cat([pooled_stocks, market_embedding], dim=-1)
        
        # Compute value
        value = self.value_network(combined)
        
        return value


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating stock embeddings.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize attention pooling.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        
        # Learned query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            batch_first=True
        )
    
    def forward(self, stock_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            stock_embeddings: Stock embeddings [batch_size, num_stocks, embedding_dim]
            
        Returns:
            Pooled embedding [batch_size, embedding_dim]
        """
        batch_size = stock_embeddings.shape[0]
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Compute attention-weighted pooling
        pooled, _ = self.attention(query, stock_embeddings, stock_embeddings)
        
        # Remove sequence dimension
        pooled = pooled.squeeze(1)  # [B, D]
        
        return pooled
