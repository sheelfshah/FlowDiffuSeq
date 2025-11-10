#!/usr/bin/env python
import jax.numpy as jnp
import flax.linen as nn


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
        x_with_t = jnp.concatenate([x_flat, t], axis=-1) # (batch, len_dim * input_dims + 1)
        
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

def get_model(args, key):
    if args.model == "MLP":
        model = MLP(
            args.len_dim,
            args.embedding_dimension,
            args.embedding_dimension
        )
    else:
        raise ValueError("Invalid model type")
    key, init_key = random.split(key)
    dummy_x = jnp.ones((1, args.len_dim, args.embedding_dimension))
    dummy_t = jnp.ones((1,))
    variables = model.init(init_key, dummy_x, dummy_t)
    return model, variables, key

    