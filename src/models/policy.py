"""
Policy network with Dirichlet distribution for portfolio generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs portfolio weights using Dirichlet distribution.
    
    Ensures valid portfolio weights (all positive, sum to 1) through the
    properties of the Dirichlet distribution.
    
    AGGRESSIVE MODE: Set aggressive=True to generate more concentrated,
    non-equal weight portfolios that make active bets.
    """
    
    def __init__(
        self,
        stock_embedding_dim: int,
        market_embedding_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_weight: float = 0.2,
        min_concentration: float = 0.1,
        aggressive: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize policy network.
        
        Args:
            stock_embedding_dim: Dimension of stock embeddings
            market_embedding_dim: Dimension of market embeddings
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            max_weight: Maximum weight for any single stock
            min_concentration: Minimum concentration parameter for Dirichlet.
                Lower values (0.01-0.1) create more diverse portfolios.
                Higher values create more uniform weight distributions.
            aggressive: If True, use settings that encourage more concentrated bets
            temperature: Temperature for scaling concentration params (lower = more concentrated)
        """
        super().__init__()
        
        self.stock_embedding_dim = stock_embedding_dim
        self.market_embedding_dim = market_embedding_dim
        self.max_weight = max_weight
        self.aggressive = aggressive
        self.temperature = temperature
        
        # AGGRESSIVE MODE: Use much lower min_concentration for peaked distributions
        if aggressive:
            self.min_concentration = 0.01  # Very low = very peaked distributions
        else:
            self.min_concentration = min_concentration
        
        # Input includes: stock embedding + market embedding + previous weight
        input_dim = stock_embedding_dim + market_embedding_dim + 1
        
        # MLP to compute score for each stock
        self.score_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Concentration parameter network (for Dirichlet distribution)
        # In aggressive mode, add more layers for better discrimination
        if aggressive:
            self.concentration_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.concentration_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(
        self,
        stock_embeddings: torch.Tensor,
        market_embedding: torch.Tensor,
        previous_weights: torch.Tensor,
        deterministic: bool = False
    ) -> tuple:
        """
        Forward pass to generate portfolio weights.
        
        Args:
            stock_embeddings: Stock embeddings [batch_size, num_stocks, stock_emb_dim]
            market_embedding: Market embedding [batch_size, market_emb_dim]
            previous_weights: Previous portfolio weights [batch_size, num_stocks]
            deterministic: If True, return mean of distribution (for evaluation)
            
        Returns:
            Tuple of (weights, log_probs, entropy, concentration_params)
        """
        batch_size, num_stocks, _ = stock_embeddings.shape
        
        # Expand market embedding to match number of stocks
        market_emb_expanded = market_embedding.unsqueeze(1).expand(-1, num_stocks, -1)
        
        # Expand previous weights
        prev_weights_expanded = previous_weights.unsqueeze(-1)
        
        # Concatenate all inputs
        combined_input = torch.cat([
            stock_embeddings,
            market_emb_expanded,
            prev_weights_expanded
        ], dim=-1)  # [B, N, stock_emb + market_emb + 1]
        
        # Compute concentration parameters for Dirichlet distribution
        concentration_logits = self.concentration_network(combined_input).squeeze(-1)
        
        if self.aggressive:
            # AGGRESSIVE MODE: Scale logits to create more differentiation
            # Use temperature to control how peaked the distribution is
            # Lower temperature = more concentrated weights
            scaled_logits = concentration_logits / self.temperature
            
            # Apply softmax-like scaling to create relative differences
            # This makes the model's predictions more impactful
            logit_range = scaled_logits.max(dim=-1, keepdim=True)[0] - scaled_logits.min(dim=-1, keepdim=True)[0]
            normalized_logits = (scaled_logits - scaled_logits.mean(dim=-1, keepdim=True)) / (logit_range + 1e-8)
            
            # Convert to concentration parameters with amplified differences
            # exp() amplifies differences: small changes in logits = large changes in concentration
            concentration_params = torch.exp(normalized_logits * 2.0) * 0.5 + self.min_concentration
        else:
            # Standard mode
            concentration_params = F.softplus(concentration_logits) + self.min_concentration
        
        # Create Dirichlet distribution
        dirichlet_dist = Dirichlet(concentration_params)
        
        if deterministic:
            # Use mean of Dirichlet distribution
            # Mean of Dirichlet(α) = α / sum(α)
            weights = concentration_params / concentration_params.sum(dim=-1, keepdim=True)
            log_probs = dirichlet_dist.log_prob(weights)
        else:
            # Sample from Dirichlet distribution
            weights = dirichlet_dist.rsample()
            log_probs = dirichlet_dist.log_prob(weights)
        
        # Apply maximum weight constraint
        weights = self._apply_max_weight_constraint(weights)
        
        # Compute entropy for exploration bonus
        entropy = dirichlet_dist.entropy()
        
        return weights, log_probs, entropy, concentration_params
    
    def _apply_max_weight_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply maximum weight constraint and renormalize.
        
        Args:
            weights: Portfolio weights [batch_size, num_stocks]
            
        Returns:
            Constrained and renormalized weights
        """
        # Clip weights to max_weight
        weights = torch.clamp(weights, 0, self.max_weight)
        
        # Renormalize to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights
    
    def get_action(
        self,
        stock_embeddings: torch.Tensor,
        market_embedding: torch.Tensor,
        previous_weights: torch.Tensor,
        deterministic: bool = False
    ) -> tuple:
        """
        Get action (portfolio weights) from the policy.
        
        Args:
            stock_embeddings: Stock embeddings
            market_embedding: Market embedding
            previous_weights: Previous portfolio weights
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (weights, log_probs, entropy)
        """
        weights, log_probs, entropy, _ = self.forward(
            stock_embeddings,
            market_embedding,
            previous_weights,
            deterministic
        )
        
        return weights, log_probs, entropy
    
    def evaluate_actions(
        self,
        stock_embeddings: torch.Tensor,
        market_embedding: torch.Tensor,
        previous_weights: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Used during PPO training to compute policy loss.
        
        Args:
            stock_embeddings: Stock embeddings
            market_embedding: Market embedding
            previous_weights: Previous portfolio weights
            actions: Actions to evaluate (portfolio weights)
            
        Returns:
            Tuple of (log_probs, entropy, concentration_params)
        """
        batch_size, num_stocks, _ = stock_embeddings.shape
        
        # Expand embeddings
        market_emb_expanded = market_embedding.unsqueeze(1).expand(-1, num_stocks, -1)
        prev_weights_expanded = previous_weights.unsqueeze(-1)
        
        # Concatenate inputs
        combined_input = torch.cat([
            stock_embeddings,
            market_emb_expanded,
            prev_weights_expanded
        ], dim=-1)
        
        # Compute concentration parameters
        concentration_logits = self.concentration_network(combined_input).squeeze(-1)
        
        if self.aggressive:
            # AGGRESSIVE MODE: Same scaling as forward pass
            scaled_logits = concentration_logits / self.temperature
            logit_range = scaled_logits.max(dim=-1, keepdim=True)[0] - scaled_logits.min(dim=-1, keepdim=True)[0]
            normalized_logits = (scaled_logits - scaled_logits.mean(dim=-1, keepdim=True)) / (logit_range + 1e-8)
            concentration_params = torch.exp(normalized_logits * 2.0) * 0.5 + self.min_concentration
        else:
            concentration_params = F.softplus(concentration_logits) + self.min_concentration
        
        # Create Dirichlet distribution
        dirichlet_dist = Dirichlet(concentration_params)
        
        # Evaluate log probability of actions
        log_probs = dirichlet_dist.log_prob(actions)
        
        # Compute entropy
        entropy = dirichlet_dist.entropy()
        
        return log_probs, entropy, concentration_params
