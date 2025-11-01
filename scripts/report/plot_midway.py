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
plt.show()
plt.savefig("bleu_rouge_vs_steps.pdf", bbox_inches="tight")
