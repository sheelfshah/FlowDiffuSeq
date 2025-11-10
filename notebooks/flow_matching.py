#!/usr/bin/env python
#  numpy 1.21.2
import os
import sys
import time
main_dir = "/home/sheels/Fall2025/10617/DiffuSeq"
os.chdir(main_dir)
sys.path.append(main_dir)
time_str = time.strftime("%Y%m%d_%H%M", time.localtime())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

from diffuseq.transformer_model import TransformerNetModel, MLP
from diffuseq.rounding import get_efficient_knn
from basic_utils import add_dict_to_argparser

from tqdm import tqdm
import argparse
import json

class Config:
    def __init__(self):
        default_config = {
            "vocab_size": 30522,
            "embedding_dimension": 128,
            "config_name": "bert-base-uncased",
            "bsz": 256,
            "num_epochs": 100,
            "model": "TransformerNet",
            "write_model": True,
            "train": False,
            "OVERFIT": False,
            "PRINT_GRAD": False
        }
        self.parser = argparse.ArgumentParser()
        add_dict_to_argparser(self.parser, default_config)
        self.args = self.parser.parse_args()
    
    def save_config(self, main_dir, time_str):
        os.makedirs(os.path.join(main_dir, f"diffusion_models/{time_str}"), exist_ok=True)
        with open(os.path.join(main_dir, f"diffusion_models/{time_str}/config.json"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)
    
    def redirect_output(self, main_dir, time_str):
        sys.stdout = open(os.path.join(main_dir, f"diffusion_models/{time_str}/output.log"), "w")
        sys.stderr = open(os.path.join(main_dir, f"diffusion_models/{time_str}/error.log"), "w")

cfg = Config()
cfg.save_config(main_dir, time_str)
cfg.redirect_output(main_dir, time_str)
args = cfg.args


device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_count = torch.cuda.device_count()
print(f"Device: {device}, GPU count: {gpu_count}")

torch.manual_seed(102)
np.random.seed(102)

def get_embedding_dir(split):
    return os.path.join(main_dir,
        f"generation_outputs/embeddings/" + 
        f"diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qqp20251030-19:19:49/{split}_embeddings/")

# Load data using memory mapping
train_x_embedding = np.load(get_embedding_dir("train") + "input_id_x_processed.npy", mmap_mode='r')
train_y_embedding = np.load(get_embedding_dir("train") + "input_id_y_processed.npy", mmap_mode='r')
train_x_encoding = np.load(get_embedding_dir("train") + "input_id_x.npy", mmap_mode='r')
train_y_encoding = np.load(get_embedding_dir("train") + "input_id_y.npy", mmap_mode='r')
valid_x_embedding = np.load(get_embedding_dir("valid") + "input_id_x_processed.npy", mmap_mode='r')
valid_y_embedding = np.load(get_embedding_dir("valid") + "input_id_y_processed.npy", mmap_mode='r')
valid_x_encoding = np.load(get_embedding_dir("valid") + "input_id_x.npy", mmap_mode='r')
valid_y_encoding = np.load(get_embedding_dir("valid") + "input_id_y.npy", mmap_mode='r')
test_x_embedding = np.load(get_embedding_dir("test") + "input_id_x_processed.npy", mmap_mode='r')
test_y_embedding = np.load(get_embedding_dir("test") + "input_id_y_processed.npy", mmap_mode='r')
test_x_encoding = np.load(get_embedding_dir("test") + "input_id_x.npy", mmap_mode='r')
test_y_encoding = np.load(get_embedding_dir("test") + "input_id_y.npy", mmap_mode='r')

if args.OVERFIT:
    train_x_embedding = train_x_embedding[:args.bsz]
    train_y_embedding = train_y_embedding[:args.bsz]
    train_x_encoding = train_x_encoding[:args.bsz]
    train_y_encoding = train_y_encoding[:args.bsz]

# Convert to torch arrays
train_x_embedding = torch.from_numpy(train_x_embedding)
train_y_embedding = torch.from_numpy(train_y_embedding)
train_x_encoding = torch.from_numpy(train_x_encoding)
train_y_encoding = torch.from_numpy(train_y_encoding)
valid_x_embedding = torch.from_numpy(valid_x_embedding)
valid_y_embedding = torch.from_numpy(valid_y_embedding)
valid_x_encoding = torch.from_numpy(valid_x_encoding)
valid_y_encoding = torch.from_numpy(valid_y_encoding)
test_x_embedding = torch.from_numpy(test_x_embedding)
test_y_embedding = torch.from_numpy(test_y_embedding)
test_x_encoding = torch.from_numpy(test_x_encoding)
test_y_encoding = torch.from_numpy(test_y_encoding)
dataset = TensorDataset(train_x_embedding, train_y_embedding, train_x_encoding, train_y_encoding)
valid_dataset = TensorDataset(valid_x_embedding, valid_y_embedding, valid_x_encoding, valid_y_encoding)
test_dataset = TensorDataset(test_x_embedding, test_y_embedding, test_x_encoding, test_y_encoding)

emb = torch.nn.Embedding(args.vocab_size, args.embedding_dimension)
pretrained_emb_state = torch.load(os.path.join(main_dir, "diffusion_models/model_emb.pth"))
emb.load_state_dict(pretrained_emb_state)

if args.model == "TransformerNet":
    model = TransformerNetModel(
        input_dims=train_x_embedding.shape[-1],
        output_dims=train_y_embedding.shape[-1],
        hidden_t_dim=train_x_embedding.shape[-1],
        dropout=0,
        config_name=args.config_name,
        vocab_size=args.vocab_size,
        init_pretrained='no',
        freeze_embeddings=True,
        freeze_rounding=True,
        embedding_weight=emb.weight,
    )
elif args.model == "MLP":
    model = MLP(train_x_embedding.shape[1], train_x_embedding.shape[-1], train_y_embedding.shape[-1])
else:
    raise ValueError("Invalid model type")
model.to(device)
print("Model created")


data_size = train_x_embedding.shape[0]
valid_data_size = valid_x_embedding.shape[0]
test_data_size = test_x_embedding.shape[0]
print(f"Data size: {data_size}, Valid data size: {valid_data_size}, Test data size: {test_data_size}")
data_loader = DataLoader(
    dataset=dataset,
    batch_size=args.bsz,
    shuffle=not args.OVERFIT, # Shuffle the data every epoch
    num_workers=0,  # Number of subprocesses to use for data loading (0 means main process)
    pin_memory=True,
    drop_last=True,
)
valid_data_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.bsz,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)
test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.bsz,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)


if args.train:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_valid_loss = float('inf')
    best_model = None
    for epoch in range(args.num_epochs):
        model.train()
        tot_loss = 0
        for i, (x, y, _, _) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            t = torch.rand(x.shape[0]).to(device)
            unsq_t = t.unsqueeze(-1).unsqueeze(-1)
            x_t = (1-unsq_t)*x + unsq_t*y
            v_t = y - x
            u_t = model(x_t, t)
            loss = criterion(u_t, v_t)
            with torch.no_grad():
                tot_loss += loss.item()
            loss.backward()
            if args.PRINT_GRAD:
                for p in model.parameters():
                    if p.grad is not None:
                        print(p.name, (p.grad ** 2).sum().item())
            optimizer.step()
        model.eval()
        valid_tot_loss = 0
        with torch.no_grad():
            for i, (x, y, _, _) in tqdm(enumerate(valid_data_loader)):
                x = x.to(device)
                y = y.to(device)
                t = torch.rand(x.shape[0]).to(device)
                unsq_t = t.unsqueeze(-1).unsqueeze(-1)
                x_t = (1-unsq_t)*x + unsq_t*y
                v_t = y - x
                u_t = model(x_t, t)
                loss = criterion(u_t, v_t)
                valid_tot_loss += loss.item()
        tqdm.write(f'Epoch {epoch}, Loss: {tot_loss/(data_size//args.bsz)}, Valid Loss: {valid_tot_loss/(valid_data_size//args.bsz)}')
        if valid_tot_loss < best_valid_loss:
            best_valid_loss = valid_tot_loss
            best_model = model.state_dict()
    if args.write_model and best_model is not None:
        print("Saving best model with valid loss: ", best_valid_loss)
        os.makedirs(os.path.join(main_dir, f"diffusion_models/{time_str}"), exist_ok=True)
        torch.save(best_model, os.path.join(main_dir, f"diffusion_models/{time_str}/model_flow.pth"))
else:
    print("Loading model from checkpoint")
    model.load_state_dict(torch.load(os.path.join(main_dir, "diffusion_models/model_flow.pth")))

print("Doing ODE solve")
x, y, x_tok, y_tok = next(iter(data_loader))
x = x.to(device)
y = y.to(device)
x_t = x.clone()
dt = 0.01
with torch.no_grad():
    model.eval()
    for t in torch.arange(0, 1, dt):
        u_t = model(x_t, torch.ones(x.shape[0]).to(device)*t)
        x_t = x_t + dt*u_t
        # print(u_t[0][0][0], x_t[0][0][0])


def round_to_nearest_token(model, text_emb):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight.clone().detach()  # input_embs
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    rounded_tokens = rounded_tokens.view(old_shape[:-1]).to(old_device)
    return new_embeds, rounded_tokens

round_pred, tok_pred = round_to_nearest_token(emb, x_t)


print("x_tok[0], tok_pred[0], y_tok[0]")
print(x_tok[0], tok_pred[0], y_tok[0])





