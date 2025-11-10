#!/usr/bin/env python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
from transformers import AutoTokenizer
import numpy as np

from tqdm import tqdm
from functools import partial
import os

from utils import main_dir, time_str, Config, get_embedding_dir
from models import MLP, get_model
from data_utils import load_embeddings, data_generator

cfg = Config(main_dir, time_str)
cfg.save_config()
cfg.redirect_output()
args = cfg.args

# JAX device setup
devices = jax.devices()
print(f"Available JAX devices: {devices}")
print(f"Device count: {len(devices)}")

# Set random seeds
np.random.seed(102)
key = random.PRNGKey(102)

# Load data using memory mapping
train_x_embedding, train_y_embedding, train_x_encoding, train_y_encoding = load_embeddings(get_embedding_dir("train"))
valid_x_embedding, valid_y_embedding, valid_x_encoding, valid_y_encoding = load_embeddings(get_embedding_dir("valid"))
test_x_embedding, test_y_embedding, test_x_encoding, test_y_encoding = load_embeddings(get_embedding_dir("test"))

if args.OVERFIT:
    train_x_embedding = train_x_embedding[:args.bsz]
    train_y_embedding = train_y_embedding[:args.bsz]
    train_x_encoding = train_x_encoding[:args.bsz]
    train_y_encoding = train_y_encoding[:args.bsz]

assert train_x_embedding.shape[1] == args.len_dim, f"Found dim 1 of data to be {train_x_embedding.shape[1]} instead of {args.len_dim}"
assert train_x_embedding.shape[-1] == args.embedding_dimension, f"Found dim -1 of data to be {train_x_embedding.shape[-1]} instead of {args.embedding_dimension}"
assert train_y_embedding.shape[-1] == args.embedding_dimension, f"Found dim -1 of data to be {train_y_embedding.shape[-1]} instead of {args.embedding_dimension}"

# Initialize model with parameters
model, variables, key = get_model(args, key)
print("Model created")


data_size = train_x_embedding.shape[0]
valid_data_size = valid_x_embedding.shape[0]
test_data_size = test_x_embedding.shape[0]
print(f"Data size: {data_size}, Valid data size: {valid_data_size}, Test data size: {test_data_size}")


# Define loss function
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

# Define training step
@jit
def train_step(state, x, y, key):
    """Single training step."""
    def loss_fn(params):
        # Sample random timesteps
        t = random.uniform(key, (x.shape[0],), minval=0.0, maxval=1.0)
        unsq_t = t[:, None, None]
        x_t = (1 - unsq_t) * x + unsq_t * y
        v_t = y - x
        u_t = model.apply({'params': params}, x_t, t)
        loss = mse_loss(u_t, v_t)
        return loss
    
    loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Define evaluation step
@jit
def eval_step(params, x, y, key):
    """Single evaluation step."""
    t = random.uniform(key, (x.shape[0],), minval=0.0, maxval=1.0)
    unsq_t = t[:, None, None]
    x_t = (1 - unsq_t) * x + unsq_t * y
    v_t = y - x
    u_t = model.apply({'params': params}, x_t, t)
    loss = mse_loss(u_t, v_t)
    return loss

if args.train:
    # Create optimizer with frozen parameters handling
    learning_rate = args.lr
    tx = optax.adam(learning_rate)
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    best_valid_loss = float('inf')
    best_params = None
    
    for epoch in range(args.num_epochs):
        # Training
        tot_loss = 0
        num_batches = 0
        
        # Split key for this epoch
        key, epoch_key = random.split(key)
        shuffle_key, train_key = random.split(epoch_key)

        batch_gen = data_generator(
            train_x_embedding,
            train_y_embedding,
            train_x_encoding,
            train_y_encoding,
            args.bsz,
            shuffle=True,
            key=shuffle_key
        )
        
        for i, (x, y, _, _) in enumerate(tqdm(batch_gen)):
            # Split key for this batch
            train_key, batch_key = random.split(train_key)
            state, loss = train_step(state, x, y, batch_key)
            tot_loss += loss
            num_batches += 1
        
        # Validation
        valid_tot_loss = 0
        valid_num_batches = 0
        
        key, valid_key = random.split(key)
        batch_gen = data_generator(
            valid_x_embedding,
            valid_y_embedding,
            valid_x_encoding,
            valid_y_encoding,
            args.bsz,
            shuffle=False,
            key=valid_key
        )
        for i, (x, y, _, _) in enumerate(batch_gen):
            valid_key, batch_key = random.split(valid_key)
            loss = eval_step(state.params, x, y, batch_key)
            valid_tot_loss += loss
            valid_num_batches += 1
        
        avg_train_loss = tot_loss / num_batches
        avg_valid_loss = valid_tot_loss / valid_num_batches
        
        tqdm.write(f'Epoch {epoch}, Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}')
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_params = state.params
    
    if args.write_model and best_params is not None:
        print(f"Saving best model with valid loss: {best_valid_loss:.6f}")
        checkpoints.save_checkpoint(
            ckpt_dir=cfg.output_dir,
            target=best_params,
            step=0,
            prefix='model_flow_',
            overwrite=True
        )
else:
    print("Loading model from checkpoint")
    checkpoint_dir = os.path.join(main_dir, "diffusion_models")
    restored_params = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=variables['params'],
        prefix='model_flow_'
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=restored_params,
        tx=optax.adam(args.lr)  # Dummy optimizer for inference
    )
