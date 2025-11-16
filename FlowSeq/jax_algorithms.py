#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from utils import main_dir, time_str, Config, get_embedding_dir, plot_loss
from models import get_model, get_embedding_matrix
from data_utils import load_embeddings, data_generator

import os

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

def save_checkpoint(epoch, params, loss, loss_t, cfg, args, force=False):
    valid_epoch = force or (epoch + 1) % args.checkpointing_interval == 0
    if args.write_model and valid_epoch and params is not None:
        print(f"Saving checkpoint at epoch {epoch + 1} with valid loss: {loss:.6f}")
        checkpoints.save_checkpoint(
            ckpt_dir=cfg.output_dir,
            target=params,
            step=epoch,
            prefix='model_flow_',
            overwrite=force,
            keep=100 # keep last 100 checkpoints
        )
        if args.importance_sampling:
            loss_t.plot(cfg.output_dir)
        plot_loss(cfg.output_dir)

class LossTracker:
    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        self.bins = jnp.linspace(0, 1, num_bins+1)
        self.sum_loss_sq = jnp.zeros(num_bins)
        self.count = jnp.zeros(num_bins)
    
    def reset(self):
        self.sum_loss_sq = jnp.zeros(self.num_bins)
        self.count = jnp.zeros(self.num_bins)
    
    def update(self, t, loss):
        """
        t:        (B,)
        loss:     (B,)
        Updates:
        sum_loss_sq_per_bin:  (num_bins,)
        count_per_bin:     (num_bins,)
        """
        assert t.shape == loss.shape
        assert t.ndim == 1

        # digitize gives bin indices 1..num_bins
        bin_idx = jnp.digitize(t, self.bins) - 1   # now 0..num_bins-1

        # -------- vectorized binning using segment_sum -------- #
        # sum of losses per bin
        self.sum_loss_sq += jax.ops.segment_sum(loss, bin_idx, self.num_bins)**2
        # how many entries fall in each bin
        self.count += jax.ops.segment_sum(jnp.ones_like(loss), bin_idx, self.num_bins)
    
    def sample_t(self, key, size):
        eps = 1e-3
        p = jnp.sqrt(self.sum_loss_sq / (self.count + 1) + eps)
        p = p / p.sum()
        random_bins = random.choice(key, self.num_bins, p=p, shape=(size,))
        t = jax.random.uniform(key, (size,)) * (self.bins[random_bins+1] - self.bins[random_bins]) + self.bins[random_bins]
        q_t = p[random_bins] / (self.bins[random_bins+1] - self.bins[random_bins])
        return t, q_t
    
    def plot(self, output_dir):
        plt.plot(self.bins[:-1], jnp.sqrt(self.sum_loss_sq / (self.count + 1)))
        plt.savefig(os.path.join(output_dir, "loss_tracker.png"))
        plt.close()

class FlowMatching:
    def __init__(self, main_dir, time_str, seed=102):
        self.main_dir = main_dir
        self.time_str = time_str
        self.cfg = Config(main_dir, time_str)
        self.cfg.save_config()
        self.cfg.redirect_output()
        self.args = self.cfg.args
        
        devices = jax.devices()
        print(f"Available JAX devices: {devices}")
        print(f"Device count: {len(devices)}")
        
        np.random.seed(seed)
        self.key = random.PRNGKey(seed)

        self.load_data()
        self.load_model()
        self.load_embedding_matrix()
        self.check_embedding_with_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.config_name)
        self.load_checkpoint()
    
    def load_data(self):
        # Load data using memory mapping
        self.train_x_embedding, self.train_y_embedding, self.train_x_encoding, self.train_y_encoding = load_embeddings(get_embedding_dir("train"))
        self.valid_x_embedding, self.valid_y_embedding, self.valid_x_encoding, self.valid_y_encoding = load_embeddings(get_embedding_dir("valid"))
        self.test_x_embedding, self.test_y_embedding, self.test_x_encoding, self.test_y_encoding = load_embeddings(get_embedding_dir("test"))

        if self.args.OVERFIT:
            self.train_x_embedding = self.train_x_embedding[:self.args.bsz]
            self.train_y_embedding = self.train_y_embedding[:self.args.bsz]
            self.train_x_encoding = self.train_x_encoding[:self.args.bsz]
            self.train_y_encoding = self.train_y_encoding[:self.args.bsz]

        assert self.train_x_embedding.shape[1] == self.args.len_dim, f"Found dim 1 of data to be {self.train_x_embedding.shape[1]} instead of {self.args.len_dim}"
        assert self.train_x_embedding.shape[-1] == self.args.embedding_dimension, f"Found dim -1 of data to be {self.train_x_embedding.shape[-1]} instead of {self.args.embedding_dimension}"
        assert self.train_y_embedding.shape[-1] == self.args.embedding_dimension, f"Found dim -1 of data to be {self.train_y_embedding.shape[-1]} instead of {self.args.embedding_dimension}"

        data_size = self.train_x_embedding.shape[0]
        valid_data_size = self.valid_x_embedding.shape[0]
        test_data_size = self.test_x_embedding.shape[0]
        print(f"Data size: {data_size}, Valid data size: {valid_data_size}, Test data size: {test_data_size}")
    
    def load_model(self):
        self.model, self.variables, self.key = get_model(self.args, self.key)
        print("Model created")
    
    def update_params(self, params):
        self.variables['params'] = params
    
    def load_checkpoint(self):
        if self.args.checkpoint_dir != "":
            print("Loading model from checkpoint")
            checkpoint_dir = os.path.join(self.main_dir, f"diffusion_models/{self.args.checkpoint_dir}")
            restored_params = checkpoints.restore_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=None, #variables['params'],
                prefix='model_flow_'
            )
            self.update_params(restored_params)
    
    def load_embedding_matrix(self):
        self.embedding_matrix = get_embedding_matrix()
        print("Embedding matrix loaded")
        assert self.embedding_matrix.shape[1] == self.args.embedding_dimension, f"Found dim 1 of embedding matrix to be {self.embedding_matrix.shape[1]} instead of {self.args.embedding_dimension}"
        assert self.embedding_matrix.shape[0] == self.args.vocab_size, f"Found dim 0 of embedding matrix to be {self.embedding_matrix.shape[0]} instead of {self.args.vocab_size}"
    
    def check_embedding_with_data(self):
        rounded_tokens, nn_idx = batch_nearest_token_rounding(self.embedding_matrix, self.train_x_embedding[:self.args.bsz])
        assert jnp.allclose(rounded_tokens, self.train_x_embedding[:self.args.bsz])
        assert jnp.allclose(nn_idx, self.train_x_encoding[:self.args.bsz])
    
    def create_schedule(self):
        return optax.warmup_exponential_decay(
            init_value=1e-7,
            peak_value=self.args.lr,
            warmup_steps=self.args.warmup_steps,
            transition_steps=self.args.transition_steps,
            decay_rate=self.args.decay_rate,
            transition_begin=self.args.transition_begin,
            staircase=False
        )
    
    def load_optimizer(self):
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.args.max_grad_norm),
            optax.scale_by_schedule(self.create_schedule()),
            optax.adam(1.)
        )
    
    def create_model_io(self, x, y, key):
        if self.args.joint_diffusion_model:
            y = jnp.concatenate([x, y], axis=1) # concat along len_dim
            noise = random.normal(key, (x.shape[0], self.args.len_dim, self.args.embedding_dimension))
            x = jnp.concatenate([x, noise], axis=1)
        return x, y
    
    def create_generator(self, train=True):
        if train:
            return data_generator(
                self.train_x_embedding, self.train_y_embedding,
                self.train_x_encoding, self.train_y_encoding,
                self.args.bsz, shuffle=True, key=self.key
            )
        else:
            return data_generator(
                self.valid_x_embedding, self.valid_y_embedding,
                self.valid_x_encoding, self.valid_y_encoding,
                self.args.bsz, shuffle=False, key=self.key
        )
    
    def mse_loss_individual(self, pred, target):
        return jnp.mean((pred - target) ** 2, axis=tuple(range(1, pred.ndim)))
    
    def mse_loss(self, pred, target):
        return jnp.mean((pred - target) ** 2)
    
    def forward_pass(self, params, x_t, t):
        """Single forward pass."""
        u_t = self.model.apply({'params': params}, x_t, t)
        return u_t

    def ode_solve(self, params, x_0, x_1, num_steps=100):
        """Solves the ODE using the forward pass."""
        self.key, key = random.split(self.key)
        x_0, x_1 = self.create_model_io(x_0, x_1, key)
        t = jnp.zeros(x_0.shape[0])
        x_t = x_0
        dt = 1 / num_steps
        for i in range(num_steps):
            x_t = x_t + self.forward_pass(params, x_t, t) * dt
            t = t + dt
            if self.args.joint_diffusion_model:
                x_t = x_t.at[:, :self.args.len_dim].set(x_0[:, :self.args.len_dim])
        return x_t
    
    def decode(self, x):
        if self.args.joint_diffusion_model:
            x = x[:, self.args.len_dim:]
        _, nn_idx = batch_nearest_token_rounding(self.embedding_matrix, x)
        return [self.tokenizer.decode(nn_idx[i]) for i in range(nn_idx.shape[0])]
        

def test_loss_tracker():
    print("Testing LossTracker")
    lt = LossTracker()
    t = random.uniform(random.PRNGKey(102), (10000,))
    loss = t * (1 - t)
    lt.update(t, loss)
    t, q_t = lt.sample_t(random.PRNGKey(102), 1000)
    plt.hist(t)
    plt.savefig("tmp.png")
    plt.close()
    plt.scatter(t, q_t)
    plt.savefig("tmp2.png")
    plt.close()

if __name__ == "__main__":
    flow_matching = FlowMatching(main_dir, time_str)
    x0 = flow_matching.train_x_embedding[:10]         
    x1 = flow_matching.train_y_embedding[:10]
    print([flow_matching.tokenizer.decode(row) for row in flow_matching.train_x_encoding[:10]])         
    print([flow_matching.tokenizer.decode(row) for row in flow_matching.train_y_encoding[:10]])         
    x1_hat = flow_matching.ode_solve(flow_matching.variables['params'], x0, x1)
    print(flow_matching.decode(x1_hat))
    