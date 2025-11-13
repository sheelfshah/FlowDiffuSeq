#!/bin/bash

conda create -n FlowSeq python=3.9
conda activate FlowSeq

pip install --upgrade "jax[cuda12]" flax optax
pip install transformers
pip install tqdm
# pip install torch=='2.8.0' --index-url https://download.pytorch.org/whl/cu129

