#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns

steps = [1,2,5,10,25,50,125,250,500,1000,2000]
bleu_values = [6.56245400962304e-05,5.945714060152421e-05,8.77340802966579e-05,6.494396251445722e-05,8.538547610352802e-05,0.00012633103207594976,0.0003651276775399245,0.0018444334707758798,0.010838288636346049,0.142284006724347,0.1862149695304189]
rouge_values = [0.00045321190729737283,0.00043125718981027605,0.0005967552088201046,0.0004482587866485119,0.0006785382807254791,0.0008139704629778862,0.002350968936830759,0.013623140358179808,0.09127995198518038,0.4643362303122878,0.531826974517107]

# different y-axes for bleu and rouge
# make x log scale
fig, ax1 = plt.subplots()
sns.set_theme(style="ticks", context="paper")
ax1.plot(steps, bleu_values, "o-", label="DiffuSeq")
ax1.set_xscale("log")
ax1.set_xlabel("Inference Steps")
ax1.set_ylabel("BLEU Score")
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(steps, rouge_values, "o-", label="DiffuSeq", color="tab:red")
ax2.set_ylabel("ROUGE Score")
ax2.legend()
# plt.show()
plt.savefig("bleu_rouge_vs_steps.pdf", bbox_inches="tight")
plt.close()

import pandas as pd

# read csv file
df_frozen_embeddings = pd.read_csv("../../diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qqp20251030-21:06:33/progress.csv")
df_trained_embeddings = pd.read_csv("../../diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qqp20251030-22:44:11/progress.csv")

fig, ax1 = plt.subplots()
sns.set_theme(style="ticks", context="paper")
ax1.plot([20*i for i in range(len(df_frozen_embeddings))], df_frozen_embeddings.loss, "-", label="Frozen Embeddings", color="tab:blue")
ax1.plot([20*i for i in range(len(df_trained_embeddings))], df_trained_embeddings.loss, "-", label="Trained Embeddings", color="tab:orange")
ax1.set_xlabel("Training Epochs")
ax1.set_ylabel("Training Loss")
ax1.set_ylim(min(df_frozen_embeddings.loss), max(df_trained_embeddings.loss)*0.97)
# add point for final loss and add value
# ax1.plot([20*len(df_frozen_embeddings)], df_frozen_embeddings.loss.iloc[-1], "o", color="tab:blue")
# ax1.plot([20*len(df_trained_embeddings)], df_trained_embeddings.loss.iloc[-1], "o", color="tab:orange")
# ax1.text(20*len(df_frozen_embeddings), df_frozen_embeddings.loss.iloc[-1], f"{df_frozen_embeddings.loss.iloc[-1]:.4f}", color="tab:blue")
# ax1.text(20*len(df_trained_embeddings), df_trained_embeddings.loss.iloc[-1], f"{df_trained_embeddings.loss.iloc[-1]:.4f}", color="tab:orange")
ax1.legend()
# plt.show()
plt.savefig("training_loss_vs_epochs.pdf", bbox_inches="tight")
plt.close()