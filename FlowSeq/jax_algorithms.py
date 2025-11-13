#!/usr/bin/env python

import jax.numpy as jnp

def nearest_token_rounding(model_emb, text_emb):
    """
    Args:
        model_emb: (vocab_size, embedding_dim)
        text_emb: (seqlen, embedding_dim)
    Returns:
        ((seqlen, embedding_dim), (seqlen,)) 
        rounded tokens and their indices
    """
    # use ||x-y||_2^2 = ||x||_2^2 + ||y||_2^2 - 2<x,y>
    # then do argmin over x to get the nearest token
    # since the argmin over x doesn't depend on ||y||_2^2, we can ignore it
    norm_model_emb = jnp.linalg.norm(model_emb, axis=-1, keepdims=True) # (vocab_size, 1)
    # norm_text_emb = jnp.linalg.norm(text_emb, axis=-1, keepdims=True) # (seqlen, 1)
    dist = norm_model_emb**2 - 2 * jnp.dot(model_emb, text_emb.T) # + norm_text_emb.T**2 # (vocab_size, seqlen)
    nn_idx = jnp.argmin(dist, axis=0) # (seqlen,)
    rounded_tokens = model_emb[nn_idx] # (seqlen, embedding_dim)
    return rounded_tokens, nn_idx

def batch_nearest_token_rounding(model_emb, text_emb):
    """
    Args:
        model_emb: (vocab_size, embedding_dim)
        text_emb: (bsz, seqlen, embedding_dim)
    Returns:
        ((bsz, seqlen, embedding_dim), (bsz, seqlen,)) 
        rounded tokens and their indices
    """
    bsz, seqlen, _ = text_emb.shape
    rounded_tokens, nn_idx = nearest_token_rounding(model_emb, text_emb.reshape(-1, text_emb.shape[-1]))
    rounded_tokens = rounded_tokens.reshape(bsz, seqlen, text_emb.shape[-1])
    nn_idx = nn_idx.reshape(bsz, seqlen)
    return rounded_tokens, nn_idx

        