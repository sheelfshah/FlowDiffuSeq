#!/usr/bin/env python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import numpy as np

from tqdm import tqdm
import os
import sys

from utils import main_dir, time_str, Config, get_embedding_dir
from models import get_model
from data_utils import load_embeddings, data_generator
from jax_algorithms import LossTracker, save_checkpoint, FlowMatching

FM = FlowMatching(main_dir, time_str)

# Define loss function
def mse_loss_individual(pred, target):
    return jnp.mean((pred - target) ** 2, axis=tuple(range(1, pred.ndim)))

def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

def get_t_q_t(key, bsz, loss_t=None):
    if loss_t is not None:
        # Sample from importance sampling distribution
        t, q_t = loss_t.sample_t(key, bsz)
    else:
        # Sample from uniform distribution
        t = random.uniform(key, (bsz,), minval=0.0, maxval=1.0)
        q_t = jnp.ones_like(t)
    return t, q_t

# Define training step
@jit
def train_step(model, state, x_0, x_1, t, q_t):
    """Single training step."""
    def loss_fn(params):
        unsq_t = t[:, None, None]
        x_t = (1 - unsq_t) * x_0 + unsq_t * x_1
        v_t = x_1 - x_0
        u_t = model.apply({'params': params}, x_t, t)
        loss = mse_loss_individual(u_t, v_t)
        loss = jnp.mean(loss * q_t)
        return loss
    
    loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Define evaluation step
@jit
def eval_step(model, params, x_0, x_1, key):
    """Single evaluation step."""
    t = random.uniform(key, (x_0.shape[0],), minval=0.0, maxval=1.0)
    unsq_t = t[:, None, None]
    x_t = (1 - unsq_t) * x_0 + unsq_t * x_1
    v_t = x_1 - x_0
    u_t = model.apply({'params': params}, x_t, t)
    loss_individual = mse_loss_individual(u_t, v_t)
    loss = jnp.mean(loss_individual)
    return loss, t, loss_individual

@jit
def eval_step_t(model, params, x_0, x_1, t):
    """Single evaluation step."""
    unsq_t = t[:, None, None]
    x_t = (1 - unsq_t) * x_0 + unsq_t * x_1
    v_t = x_1 - x_0
    u_t = model.apply({'params': params}, x_t, t)
    loss = mse_loss(u_t, v_t)
    return loss

@jit
def forward_pass(model, params, x_t, t):
    """Single forward pass."""
    u_t = model.apply({'params': params}, x_t, t)
    return u_t


if FM.args.train:
    # Create optimizer with frozen parameters handling
    tx = optax.adam(FM.args.lr)
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=FM.model.apply,
        params=FM.variables['params'],
        tx=tx
    )
    
    best_valid_loss = float('inf')
    best_params = None

    loss_t = LossTracker() if FM.args.importance_sampling else None
    
    for epoch in range(FM.args.num_epochs):
        # Training
        tot_loss = 0
        num_batches = 0
        
        # Split key for this epoch
        key, train_key = random.split(key)

        batch_gen = FM.create_generator(train=True)
        
        for i, (x, y, _, _) in enumerate(tqdm(batch_gen)):
            # Split key for this batch
            train_key, batch_key = random.split(train_key)
            x_0, x_1 = FM.create_model_io(x, y, batch_key)
            t, q_t = get_t_q_t(batch_key, x_0.shape[0], loss_t)
            state, loss = train_step(FM.model, state, x_0, x_1, t, q_t) 
            tot_loss += loss
            num_batches += 1
        
        # Validation
        valid_tot_loss = 0
        valid_num_batches = 0
        
        key, valid_key = random.split(key)
        batch_gen = FM.create_generator(train=False)
        if FM.args.importance_sampling:
            loss_t.reset()
        for i, (x, y, _, _) in enumerate(batch_gen):
            valid_key, batch_key = random.split(valid_key)
            x_0, x_1 = FM.create_model_io(x, y, batch_key) 
            loss, t, loss_individual = eval_step(FM.model, state.params, x_0, x_1, batch_key)
            if FM.args.importance_sampling:
                loss_t.update(t, loss_individual)
            valid_tot_loss += loss
            valid_num_batches += 1
        
        avg_train_loss = tot_loss / num_batches
        avg_valid_loss = valid_tot_loss / valid_num_batches
        
        tqdm.write(f'Epoch {epoch}, Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}')
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_params = state.params
        
        save_checkpoint(epoch, state.params, avg_valid_loss, loss_t, cfg, args)
    
    save_checkpoint(epoch, best_params, best_valid_loss, loss_t, cfg, args, force=True)

sys.exit()

# batch_gen = data_generator(
#     train_x_embedding,
#     train_y_embedding,
#     train_x_encoding,
#     train_y_encoding,
#     args.bsz,
#     shuffle=False,
#     key=random.PRNGKey(102)
# )

# (x, y, x_tok, y_tok) = next(batch_gen)
# x_0, x_1 = create_model_io(x, y, args, random.PRNGKey(102))
# print(eval_step(state.params, x_0, x_1, random.PRNGKey(102)))
# t = random.uniform(random.PRNGKey(102), (x.shape[0],))
# print(eval_step_t(state.params, x_0, x_1, t))
# t = jnp.linspace(0, 1, x.shape[0])
# lin = []
# const = []
# for i in range(256):
#     lin.append(eval_step_t(state.params, x_0, x_1, (t+i/256)%1.0))
#     const.append(eval_step_t(state.params, x_0, x_1, (i/256)*jnp.ones((x.shape[0],))))
# import matplotlib.pyplot as plt
# plt.plot(lin)
# plt.plot(const)
# plt.savefig("tmp.png")
# plt.show()
# # sys.exit()
# f1 = (forward_pass(state.params, x_0, jnp.zeros((x_0.shape[0],))))
# f2 = (forward_pass(state.params, x_0, t))
# print(mse_loss(f1[0], f2[0]))
# # sys.exit()
# print(eval_step_t(state.params, x_0, x_1, jnp.zeros((x_0.shape[0],))))
# print(eval_step_t(state.params, x_0, x_1, jnp.ones((x_0.shape[0],))))
# print(eval_step_t(state.params, x_0, x_1, jnp.ones((x_0.shape[0],))*0.5))
# # sys.exit()
# t = jnp.zeros((x_0.shape[0],))
# dt = 0.01
# x_t = x_0.copy()
# v_t = x_1 - x_0
# for i in range(int(1/dt)):
#     u_t = forward_pass(state.params, x_t, t)
#     x_t = x_t + dt * u_t
#     t = t + dt
#     print(mse_loss(u_t[0], v_t[0]), mse_loss(u_t, v_t))

# print(jnp.linalg.norm(x_t-x_1, axis=-1)[0])
# print(jnp.linalg.norm(x_t-x_0, axis=-1)[0])