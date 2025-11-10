#!/usr/bin/env python
import jax.numpy as jnp
from jax import random

# JAX data loader (simple batching)
def data_generator(x_emb, y_emb, x_enc, y_enc, batch_size, shuffle=False, key=None):
    """Generate batches of data.
    Always drops the last batch if it is not full, because jax prefers to have fixed batch size.
    """
    n_samples = x_emb.shape[0]
    indices = jnp.arange(n_samples)
    
    if shuffle and key is not None:
        indices = random.permutation(key, indices)
    
    for start_idx in range(0, n_samples - batch_size + 1, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield (
            x_emb[batch_indices],
            y_emb[batch_indices],
            x_enc[batch_indices],
            y_enc[batch_indices]
        )

def load_embeddings(folder):
    x_embedding = np.load(folder + "/input_id_x_processed.npy", mmap_mode='r')
    y_embedding = np.load(folder + "/input_id_y_processed.npy", mmap_mode='r')
    x_encoding = np.load(folder + "/input_id_x.npy", mmap_mode='r')
    y_encoding = np.load(folder + "/input_id_y.npy", mmap_mode='r')
    return jnp.array(x_embedding), jnp.array(y_embedding), jnp.array(x_encoding), jnp.array(y_encoding)