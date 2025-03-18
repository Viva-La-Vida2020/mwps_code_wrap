# Math Word Peoblems (MWPs) Code Wrap

In this project, we provide the code for several popular contrastive learning (CL) frameworks designed for Math Word Problems (MWPs). For the Encoder-Decoder model, we include BERT-GTS as a baseline and introduce Textual-CL for comparison. For the Decoder-Only model, we incorporate Galactica and SmolLM, evaluating various CL methods with them. The models are trained on Math23K and MathQA. For each model, we provide both training and testing code, along with a simple demo demonstrating how the pretrained BERT-GTS model can be used to solve user-input math problems.

## Environment Setup

This project uses Python 3.10. Follow the steps below to set up the project environment:

### Install Dependencies

First, clone the repository and navigate to the project directory:

```bash
conda create --name swip python=3.10
conda activate simpler
cd PROJECT_ROOT_PATH
pip install -r requirements.txt
```
### Train Encoder-Decoder Model from Scratch
```bash
cd PROJECT_ROOT_PATH
python -m scripts.gts.train --save_dir experiments/GTS_Mathqa --dataset mathqa --pretrained_model bert-base-uncased --devices 0
```
### Test Encoder-Decoder Model
```bash
cd PROJECT_ROOT_PATH
python -m scripts.gts.train --test --save_dir experiments/GTS_Mathqa --dataset mathqa --pretrained_model bert-base-uncased  --ckpt experiments/GTS_Mathqa/lightning_logs/version_0/checkpoints/last.ckpt --devices 0
```
### Training Decoder-only Model from Scratch
```bash
cd PROJECT_ROOT_PATH/llms
python train_decoder_only_cl.py --train --CL SimplerCL --similarity TLWD
```
### Reference Demo
```bash
cd PROJECT_ROOT_PATH
python -m scripts.gts.inference_demo --pretrained_model bert-base-uncased --ckpt /experiments/GTS_Mathqa/lightning_logs/version_0/checkpoints/last.ckpt --device 0
```
After the model is loaded, enter the question
```bash
Enter a math problem (type 'exit' to quit):
a shopkeeper sold an article offering a discount of 5 % and earned a profit of 31 % . what would have been the percentage of profit earned if no discount had been offered ?
📌 Predicted Infix Expression: ((100 * (100 + 31.0)) / (100 - 5.0)) - 100
📌 Predicted Prefix Expression: ['-', '/', '*', '+', '100', 31.0, '100', '-', '100', 5.0, '100']
🔢 Predicted Answer: 37.89473684210526

Enter a math problem (type 'exit' to quit):
exit
```