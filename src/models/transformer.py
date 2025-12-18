"""
Transformer-based stock feature encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StockFeatureEncoder(nn.Module):
    """
    Transformer encoder for stock features.
    
    Processes N stocks with F features each, producing contextual embeddings
    that capture relationships between stocks through self-attention.
    """
    
    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_sector_embedding: bool = False,
        num_sectors: int = 10
    ):
        """
        Initialize transformer encoder.
        
        Args:
            num_features: Number of input features per stock (F)
            embedding_dim: Dimension of embeddings (D)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            use_sector_embedding: Whether to use sector embeddings
            num_sectors: Number of sectors (if using sector embeddings)
        """
        super().__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Feature projection layer
        self.feature_projection = nn.Linear(num_features, embedding_dim)
        
        # Optional sector embedding
        self.use_sector_embedding = use_sector_embedding
        if use_sector_embedding:
            self.sector_embedding = nn.Embedding(num_sectors, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        stock_features: torch.Tensor,
        sector_ids: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            stock_features: Input features [batch_size, num_stocks, num_features]
            sector_ids: Sector IDs [batch_size, num_stocks] (optional)
            mask: Attention mask [batch_size, num_stocks] (optional)
            
        Returns:
            Stock embeddings [batch_size, num_stocks, embedding_dim]
        """
        batch_size, num_stocks, _ = stock_features.shape
        
        # Project features to embedding space
        x = self.feature_projection(stock_features)  # [B, N, D]
        
        # Add sector embeddings if available
        if self.use_sector_embedding and sector_ids is not None:
            sector_emb = self.sector_embedding(sector_ids)  # [B, N, D]
            x = x + sector_emb
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        # Self-attention learns correlation matrix between stocks
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Layer normalization
        encoded = self.layer_norm(encoded)
        
        return encoded  # [B, N, D]


class MarketContextEncoder(nn.Module):
    """
    MLP encoder for market-level features.
    """
    
    def __init__(
        self,
        num_market_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize market context encoder.
        
        Args:
            num_market_features: Number of market features (K)
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_market_features = num_market_features
        self.embedding_dim = embedding_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(num_market_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            market_features: Market features [batch_size, num_market_features]
            
        Returns:
            Market embedding [batch_size, embedding_dim]
        """
        x = self.mlp(market_features)
        x = self.layer_norm(x)
        return x


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
        
        # Learned query vector
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Attention weights computation
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
        
        # Compute attention
        pooled, _ = self.attention(query, stock_embeddings, stock_embeddings)
        
        # Remove sequence dimension
        pooled = pooled.squeeze(1)  # [B, D]
        
        return pooled
