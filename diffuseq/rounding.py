import torch
# bert results
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np

def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices

def denoised_fn_round(args, model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
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

    return new_embeds


# def compute_logp(args, model, x, input_ids):
#     word_emb = model.weight
#     sigma = 0.1
#     if args.model_arch == '1d-unet':
#         x = x.permute(0, 2, 1)

#     bsz, seqlen, dim = x.shape

#     x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
#     word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
#     diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

#     logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
#     logp_expanded = logp_expanded.permute((1, 0))

#     ce = torch.nn.CrossEntropyLoss(reduction='none')
#     loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

#     return loss
