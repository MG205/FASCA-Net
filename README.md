# FASCA-Net
This repository implements FASCA-Net, a multimodal architecture for affect/sentiment recognition that (i) performs fine-grained image–text and speech–text alignment, and (ii) fuses text, audio, and visual streams with cross-modal attention plus intra-modal self-attention. This README provides replication-ready instructions to set up the environment, obtain/prepare data, run training/evaluation, and reproduce the reported experiments end-to-end.

# 1. What’s in this repo

FASCA-Net/

├─ data/                # Place or auto-download datasets & processed features here

├─ models/              # Model components: encoders, FGFA_TA, fusion blocks, heads

├─ tools/               # Data prep, feature extraction, alignment, evaluation scripts

├─ checkpoint/          # Saved weights and logs (created at runtime)

├─ main.py              # Entry point: training/eval CLI

├─ requirements.txt     # Python dependencies

└─ README.md            # This file



# 2. Method overview (architecture at a glance)
## 2.1 Multimodal fusion (Text–Audio–Vision)

Dimensionality alignment. Raw features from text, speech, and vision are projected by a per-modality linear layer to a shared hidden space.
Cross-modal attention. For each modality (as Query), we concatenate the other two modalities as Key/Value and apply multi-head attention to capture complementary cues across streams.
Intra-modal encoding. A standard self-attention block then strengthens within-modality context and semantic consistency.

Residual de-projection. The fused hidden states are linearly projected back to each modality’s native space and residually added to the inputs (scale/shape preserved), yielding aligned & enriched representations for downstream classification/regression.

## 2.2 Fine-Grained Feature Alignment (FGFA_TA)

Pairwise alignment (speech↔text, image↔text). We first project the paired modalities to the same dimension, build dual-branch fused feature maps, and apply reference-based feature enrichment (RBFE) to address temporal/frame misalignments (inspired by burst alignment in vision).

FARModules with deformable conv. We predict offsets and masks to guide DeformConv2d for precise time-axis resampling, then expand the reference features and apply dual residual fusion to refine alignment and denoise residual jitters.

Output. The aligned features preserve each modality’s semantics yet are finely synchronized for downstream heads, generalizing to other 1D/2D cross-modal alignment tasks.

# 3. Replicable research: datasets & preparation
We evaluate on standard public benchmarks for multimodal affect/sentiment and emotion recognition. We do not redistribute copyrighted data; instead, please obtain them from their owners and prepare as below.

## 3.1 Datasets (obtain from official sources)
CMU-MOSI — opinion video clips with sentiment annotations (text/audio/vision). Download & details: Multicomp Lab page. 
CMU-MOSEI — 23k+ segments, 1,000 speakers, sentiment + emotions; distributed via the CMU Multimodal (Affect) Data SDK. 
IEMOCAP — acted dyadic emotion database (audio/vision/text). Request access from USC SAIL. 
MELD — multimodal, multi-party emotions in conversations (Friends TV series). Project & GitHub. 

We rely on the CMU Multimodal Data SDK (MMSDK) to download/align MOSI/MOSEI feature files and splits when applicable


## 3.2 Directory layout

After you download/request the datasets, organize them as:
data/
  mosi/
    raw/ ...             # raw media and/or official .csd feature files
    processed/ ...       # cached tensors created by tools/*
  mosei/
    raw/ ...
    processed/ ...
  iemocap/
    raw/ ...
    processed/ ...
  meld/
    raw/ ...
    processed/ ...


## 3.3 Feature extraction & alignment

We support two routes:

#### Route A (recommended for MOSI/MOSEI): use MMSDK .csd features (OpenFace facial, COVAREP acoustic, timestamped word vectors) and official splits; then run our preprocessing to convert .csd to model-ready tensors. See MMSDK docs and community tutorials. 

#### Route B: run our scripts in tools/ to extract visual (OpenFace or facenet), acoustic (librosa/COVAREP-like), and textual (transformer tokenizer) features directly from raw media.


- Example: prepare CMU-MOSEI using MMSDK exports

python tools/prepare_mosei.py \
  --mmsdk_root /path/to/mmsdk/features \
  --out_dir data/mosei/processed \
  --splits standard

- Example: prepare MELD from raw media with our extractors

python tools/extract_meld.py \
  --meld_root data/meld/raw \
  --out_dir data/meld/processed \
  --n_jobs 8




# 4. Installation

We recommend a fresh virtual environment.

- clone

git clone https://github.com/MG205/FASCA-Net.git

cd FASCA-Net

- create environment (Python 3.10+ recommended)
  
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

- install dependencies
  
pip install --upgrade pip

pip install -r requirements.txt





# 5. Quick start (inference)
Assuming you have a prepared sample (text tokens, acoustic/visual features) in data/sample/:

# show CLI
python main.py --help

# run a single forward pass with a provided checkpoint
python main.py \
  --mode infer \
  --input_dir data/sample \
  --ckpt checkpoint/fasca_net_best.pt \
  --save_pred pred.json



# 6.  Training & evaluation (full reproduction)


Below are drop-in commands to reproduce typical experiments on each dataset. All runs set the same random seed to ensure replicability.



## 6.1 CMU-MOSEI (sentiment regression + 7-class emotion)

# Train

python main.py \
  --mode train \
  --dataset mosei \
  --data_dir data/mosei/processed \
  --modalities text audio vision \
  --model fasca_net \
  --epochs 50 --batch_size 32 --lr 2e-4 \
  --seed 2025 \
  --save_dir checkpoint/mosei_fasca


# Evaluate (MAE/RMSE/Acc/F1 etc.)

python main.py \
  --mode eval \
  --dataset mosei \
  --data_dir data/mosei/processed \
  --ckpt checkpoint/mosei_fasca/best.pt \
  --eval_metrics mae rmse acc f1 auc


## 6.2 CMU-MOSI (binary/5-class sentiment)


python main.py --mode train \
  --dataset mosi --data_dir data/mosi/processed \
  --modalities text audio vision \
  --epochs 50 --batch_size 32 --lr 2e-4 --seed 2025 \
  --save_dir checkpoint/mosi_fasca

python main.py --mode eval \
  --dataset mosi --data_dir data/mosi/processed \
  --ckpt checkpoint/mosi_fasca/best.pt \
  --eval_metrics acc f1 auc mae


## 6.3 IEMOCAP (emotion classification)


python main.py --mode train \
  --dataset iemocap --data_dir data/iemocap/processed \
  --modalities text audio vision \
  --epochs 60 --batch_size 16 --lr 3e-4 --seed 2025 \
  --save_dir checkpoint/iemocap_fasca


python main.py --mode eval \
  --dataset iemocap --data_dir data/iemocap/processed \
  --ckpt checkpoint/iemocap_fasca/best.pt \
  --eval_metrics acc f1 auc

## 6.4 MELD (emotion in conversations)


python main.py --mode train \
  --dataset meld --data_dir data/meld/processed \
  --modalities text audio vision \
  --epochs 40 --batch_size 32 --lr 2e-4 --seed 2025 \
  --save_dir checkpoint/meld_fasca


python main.py --mode eval \
  --dataset meld --data_dir data/meld/processed \
  --ckpt checkpoint/meld_fasca/best.pt \
  --eval_metrics acc f1 auc
