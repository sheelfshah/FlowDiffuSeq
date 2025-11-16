#!/usr/bin/env python
import os
import sys
import time
main_dir = "/home/sheels/Fall2025/10617/DiffuSeq/FlowSeq/"
os.chdir(main_dir)
sys.path.append(main_dir)
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


class Config:
    def __init__(self, main_dir, time_str):
        default_config = {
            "vocab_size": 30522,
            "embedding_dimension": 128,
            "config_name": "bert-base-uncased",
            "bsz": 256,
            #optimizer
            "num_epochs": 100,
            "lr": 1e-4,
            "max_grad_norm": 1.,
            "warmup_steps": 1000,
            "transition_steps": 50000,
            "decay_rate": 0.9,
            "transition_begin": 200000,

            "model": "MLP",
            "len_dim": 64,
            "write_model": True,
            "train": False,
            "checkpoint_dir": "",
            "checkpointing_interval": 100000,
            "joint_diffusion_model": False,
            "importance_sampling": False,
            "model__num_layers": 2,
            "model__num_heads": 4,
            "model__mlp_ratio": 2,
            "model__dropout_rate": 0.,
            "model__hidden_dim": 256,
            "model__time_emb_dim": 256,
            "OVERFIT": False,
            "PRINT_GRAD": False,
            "NO_REDIRECT": False
        }
        self.parser = argparse.ArgumentParser()
        add_dict_to_argparser(self.parser, default_config)
        self.args = self.parser.parse_args()
        self.main_dir = main_dir
        self.time_str = time_str
        self.output_dir = os.path.join(self.main_dir, f"diffusion_models/{self.time_str}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_config(self):
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)
    
    def redirect_output(self):
        if not self.args.NO_REDIRECT:
            sys.stdout = open(os.path.join(self.output_dir, "output.log"), "w")
            sys.stderr = open(os.path.join(self.output_dir, "error.log"), "w")


def get_embedding_dir(split):
    return os.path.join(main_dir, f"embeddings/{split}_embeddings/")

def plot_loss(output_dir):
    ls = open(os.path.join(output_dir, "output.log")).readlines()
    ls = [l for l in ls if "Epoch" in l]
    ls = [l.replace(",", "").split() for l in ls]
    ls = [list(map(float, [l[1], l[3], l[6]])) for l in ls]
    ls = np.array(ls)
    plt.plot(ls[:, 1], label="Train")
    plt.plot(ls[:, 2], label="Valid")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()