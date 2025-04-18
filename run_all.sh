#!/bin/bash

# ========== GTS ==========
echo ">>> Running GTS"
python main.py --mode train --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_math23k_bert

python main.py --mode test --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_math23k_bert \
    --ckpt experiments/gts_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

# ========== GTS + Simpler ==========
echo ">>> Running GTS + SimplerCL"
python main.py --mode train --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_simpler_math23k_bert

python main.py --mode test --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_simpler_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_simpler_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

# ========== GTS + TextualCL ==========
echo ">>> Running GTS + TextualCL"
python main.py --mode train --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_textual_tlwd_math23k_bert --similarity tlwd

python main.py --mode test --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_textual_tlwd_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_textual_tlwd_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

# ========== LLM ==========
echo ">>> Running LLM (Galactica)"
python main.py --mode train --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_math23k

python main.py --mode test --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_math23k/checkpoints/xxx.ckpt

python main.py --mode demo --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_math23k/checkpoints/xxx.ckpt

# ========== LLM + SimplerCL ==========
echo ">>> Running LLM + SimplerCL"
python main.py --mode train --model llm_simpler --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_simpler_math23k

python main.py --mode test --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_simpler_math23k/checkpoints/xxx.ckpt

python main.py --mode demo --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_simpler_math23k/checkpoints/xxx.ckpt

# ========== LLM + ContraCLM ==========
echo ">>> Running LLM + ContraCLM"
python main.py --mode train --model llm_contraclm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_contraclm_math23k

python main.py --mode test --model llm_contraclm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_contraclm_math23k/checkpoints/xxx.ckpt

python main.py --mode demo --model llm_contraclm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_contraclm_math23k/checkpoints/xxx.ckpt