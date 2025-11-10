
# """
# Utility functions for neural networks in JAX.
# """
# import jax.numpy as jnp
# import flax.linen as nn
# import math


# def timestep_embedding(timesteps, dim, max_period=10000):
#     """
#     Create sinusoidal timestep embeddings.
    
#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     half = dim // 2
#     freqs = jnp.exp(
#         -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
#     )
#     args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
#     embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
#     if dim % 2:
#         embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
#     return embedding


# # SiLU activation is already available in flax as nn.silu
# # But keeping this for compatibility
# def silu(x):
#     """
#     SiLU (Swish) activation function.
#     """
#     return nn.silu(x)


# # Linear layer is just nn.Dense in Flax
# def linear(in_features, out_features, use_bias=True):
#     """
#     Create a linear (Dense) layer.
#     This is a wrapper for compatibility with PyTorch-style code.
#     """
#     return nn.Dense(features=out_features, use_bias=use_bias)
#

from transformers import AutoConfig
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Optional, Any
import numpy as np

from .utils.nn import (
    timestep_embedding,
)

class MLP(nn.Module):
    """MLP model for flow matching."""
    len_dim: int
    input_dims: int
    output_dims: int
    
    @nn.compact
    def __call__(self, x, t):
        # Flatten x and concatenate with t
        t = jnp.expand_dims(t, axis=-1)  # (batch, 1)
        x_flat = jnp.reshape(x, (x.shape[0], -1))  # (batch, len_dim * input_dims)
        x_with_t = jnp.concatenate([x_flat, t], axis=-1)
        
        # Calculate dimensions
        in_dim = self.len_dim * self.input_dims
        hidden_dim = in_dim * 1
        out_dim = self.len_dim * self.output_dims
        
        # MLP layers
        h = nn.Dense(hidden_dim)(x_with_t)
        h = nn.silu(h)
        h = nn.Dense(hidden_dim)(h)
        h = nn.silu(h)
        h = nn.Dense(out_dim)(h)
        
        # Reshape to output
        out = jnp.reshape(h, (x.shape[0], self.len_dim, self.output_dims))
        return out


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    :param freeze_embeddings: whether to freeze embedding layers
    :param embedding_weight: pretrained embedding weights
    """
    
    input_dims: int
    output_dims: int
    hidden_t_dim: int
    dropout: float = 0.0
    config_name: str = 'bert-base-uncased'
    vocab_size: int = 30522
    init_pretrained: str = 'no'
    logits_mode: int = 1
    freeze_embeddings: bool = False
    freeze_rounding: bool = False
    embedding_weight: Optional[jnp.ndarray] = None
    
    def setup(self):
        # Load config
        config = AutoConfig.from_pretrained(self.config_name)
        config.hidden_dropout_prob = self.dropout
        self.hidden_size = config.hidden_size
        
        # Time embedding network
        time_embed_dim = self.hidden_t_dim * 4
        self.time_embed_layers = [
            nn.Dense(time_embed_dim),
            nn.Dense(self.hidden_size)
        ]
        
        # Embedding layers
        if self.embedding_weight is not None:
            # Initialize with pretrained weights
            self.word_embedding = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.input_dims,
                embedding_init=lambda key, shape, dtype: self.embedding_weight
            )
        else:
            self.word_embedding = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.input_dims
            )
        
        # LM head (shares weights with embeddings)
        self.lm_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_init=lambda key, shape, dtype: self.embedding_weight.T if self.embedding_weight is not None else nn.initializers.normal()(key, shape, dtype)
        )
        
        # Input projection if needed
        if self.input_dims != self.hidden_size:
            self.input_up_proj = [
                nn.Dense(self.hidden_size),
                nn.Dense(self.hidden_size)
            ]
        
        # Position embeddings
        self.position_embeddings = nn.Embed(
            num_embeddings=config.max_position_embeddings,
            features=self.hidden_size
        )
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(epsilon=config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(rate=config.hidden_dropout_prob)
        
        # Transformer encoder
        # Note: For full BERT support, you'd use FlaxBertEncoder here
        # For now, implementing a simplified transformer
        self.num_layers = config.num_hidden_layers
        self.transformer_blocks = [
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_rate=config.hidden_dropout_prob,
                intermediate_size=config.intermediate_size
            ) for _ in range(self.num_layers)
        ]
        
        # Output projection if needed
        if self.output_dims != self.hidden_size:
            self.output_down_proj = [
                nn.Dense(self.hidden_size),
                nn.Dense(self.output_dims)
            ]
    
    def get_embeds(self, input_ids):
        """Get word embeddings for input_ids."""
        return self.word_embedding(input_ids)
    
    def get_logits(self, hidden_repr):
        """Get logits from hidden representations."""
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        else:
            raise NotImplementedError
    
    def __call__(self, x, timesteps, train: bool = False):
        """
        Apply the model to an input batch.

        :param x: an [N x L x C] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param train: whether in training mode (for dropout)
        :return: an [N x L x C] Tensor of outputs.
        """
        batch_size, seq_length, _ = x.shape
        
        # Time embedding
        emb_t = timestep_embedding(timesteps, self.hidden_t_dim)
        for layer in self.time_embed_layers[:-1]:
            emb_t = layer(emb_t)
            emb_t = nn.silu(emb_t)
        emb_t = self.time_embed_layers[-1](emb_t)
        
        # Input projection
        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj[0](x)
            emb_x = jnp.tanh(emb_x)
            emb_x = self.input_up_proj[1](emb_x)
        else:
            emb_x = x
        
        # Position embeddings
        position_ids = jnp.arange(seq_length)[None, :]  # (1, seq_length)
        position_emb = self.position_embeddings(position_ids)
        
        # Combine embeddings
        emb_t_expanded = jnp.expand_dims(emb_t, axis=1)  # (batch, 1, hidden)
        emb_t_expanded = jnp.tile(emb_t_expanded, (1, seq_length, 1))  # (batch, seq_length, hidden)
        emb_inputs = position_emb + emb_x + emb_t_expanded
        
        # Layer norm and dropout
        emb_inputs = self.layer_norm(emb_inputs)
        emb_inputs = self.dropout_layer(emb_inputs, deterministic=not train)
        
        # Transformer blocks
        hidden_states = emb_inputs
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, deterministic=not train)
        
        # Output projection
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj[0](hidden_states)
            h = jnp.tanh(h)
            h = self.output_down_proj[1](h)
        else:
            h = hidden_states
        
        return h


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    hidden_size: int
    num_heads: int
    dropout_rate: float
    intermediate_size: int
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Self-attention with residual
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )(x)
        x = x + attn_output
        x = nn.LayerNorm()(x)
        
        # Feed-forward with residual
        ff_output = x
        ff_output = nn.Dense(self.intermediate_size)(ff_output)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.hidden_size)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(ff_output)
        x = x + ff_output
        x = nn.LayerNorm()(x)
        
        return x