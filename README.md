# 🧠 Math Word Problems (MWPs) Contrastive Learning Framework

This repository provides implementations of several contrastive learning (CL) frameworks designed for solving **Math Word Problems (MWPs)**. We support both **encoder-decoder** and **decoder-only** models, including:

- ✅ **BERT-GTS** baseline  
- ✅ **Textual-CL**, **SimplerCL** for encoder-decoder  
- ✅ **Galactica** and **SmolLM** for decoder-only  
- ✅ Full support for **Math23K** and **MathQA**

Each model includes training, testing, and demo modes. A live interactive demo allows users to input math questions and get symbolic + numeric solutions.

---

## 📦 Environment Setup

We recommend using Python 3.10 and conda.

```bash
conda create --name mwps python=3.10
conda activate mwps
cd PROJECT_ROOT_PATH
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### ✅ Train Encoder-Decoder (GTS)

```bash
python main.py --mode train --model gts --dataset mathqa \
    --pretrained_model bert-base-uncased \
    --save_path experiments/gts_mathqa --devices 0
```

### ✅ Test Encoder-Decoder

```bash
python main.py --mode test --model gts --dataset mathqa \
    --pretrained_model bert-base-uncased \
    --save_path experiments/gts_mathqa \
    --ckpt experiments/gts_mathqa/lightning_logs/version_0/checkpoints/last.ckpt
```

### ✅ Train Decoder-Only (LLM)

```bash
python main.py --mode train --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_math23k
```

---

## 💡 Demo (Interactive Inference)

Run demo mode with a trained checkpoint:

```bash
python main.py --mode demo --model gts --dataset mathqa \
    --pretrained_model bert-base-uncased \
    --ckpt experiments/gts_mathqa/lightning_logs/version_0/checkpoints/last.ckpt
```

Example interaction:

```text
Enter a math problem (type 'exit' to quit):
a shopkeeper sold an article offering a discount of 5% and earned a profit of 31%. What would have been the percentage of profit earned if no discount had been offered?

📌 Predicted Infix Expression: ((100 * (100 + 31.0)) / (100 - 5.0)) - 100  
📌 Predicted Prefix Expression: ['-', '/', '*', '+', '100', 31.0, '100', '-', '100', 5.0, '100']  
🔢 Predicted Answer: 37.89...

Enter a math problem (type 'exit' to quit):
exit
```

---

## 🧪 Full Script Reference

<details>
<summary>▶ Click to expand all supported command examples</summary>

### 🔹 GTS

```bash
python main.py --mode train --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_math23k_bert

python main.py --mode test --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_math23k_bert \
    --ckpt experiments/gts_math23k_bert/lightning_logs/version_1/checkpoints/last.ckpt

python main.py --mode demo --model gts --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_math23k_bert/lightning_logs/version_1/checkpoints/last.ckpt
```

### 🔹 GTS + SimplerCL

```bash
python main.py --mode train --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_simpler_math23k_bert

python main.py --mode test --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_simpler_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model simpler --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_simpler_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt
```

### 🔹 GTS + TextualCL

```bash
python main.py --mode train --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --save_path experiments/gts_textual_tlwd_math23k_bert --similarity tlwd

python main.py --mode test --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_textual_tlwd_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model textual --dataset math23k \
    --pretrained_model pretrained_model/chinese-bert-wwm-ext \
    --ckpt experiments/gts_textual_tlwd_math23k_bert/lightning_logs/version_0/checkpoints/last.ckpt
```

### 🔹 LLM (Galactica)

```bash
python main.py --mode train --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_math23k

python main.py --mode test --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_math23k/lightning_logs/version_0/checkpoints/last.ckpt

python main.py --mode demo --model llm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_math23k/checkpoints/model-epoch=13-val_acc=0.02.ckpt
```

### 🔹 LLM + SimplerCL / ContraCLM

```bash
python main.py --mode train --model llm_simpler --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_simpler_math23k

python main.py --mode train --model llm_contraclm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --save_path experiments/gts_llm_contraclm_math23k

python main.py --mode demo --model llm_contraclm --dataset math23k \
    --pretrained_model facebook/galactica-125m \
    --ckpt experiments/gts_llm_contraclm_math23k/checkpoints/model-epoch=26-val_acc=0.18.ckpt
```

</details>

---