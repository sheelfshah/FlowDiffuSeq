#!/bin/bash

git clone https://github.com/Shark-NLP/DiffuSeq.git
cd DiffuSeq
conda create -n DiffuSeq python=3.9
conda activate DiffuSeq

pip install torch=='1.9.0' --index-url https://download.pytorch.org/whl/cu111
sed -i 's/torch==.*/torch==1.9.0+cu111/' requirements.txt
sed -i 's/numpy.*/numpy==1.21.2/' requirements.txt
pip install -r requirements.txt

pip install gdown
gdown --folder 1vnhJIUqPQva_x_sH2h5a0moCc1NYmEpr -O ./pretrained/QQP
gdown --folder 1BHGCeHRZU7MQF3rsqXBIOCU2WIC3W6fb -O ./datasets/QQP

pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"