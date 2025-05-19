"""
Main script for training, testing, and demo of Math Word Problem Solvers.

Supports:
- Encoder-Decoder models (GTS, Simpler, TextualCL)
- Decoder-only models (LLM, LLM-Simpler, LLM-ContraCLM)

"""

import argparse
import re

import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

# === Data Modules ===
import data.data_module_simpler as DataModuleSimpler
import data.data_module_gts as DataModuleGTS
import data.data_module_llm as DataModuleLLM
import data.data_module_triplet as DataModuleTriplet

# === Model Components ===
from src.models.encoder import Encoder
from src.models.tree_decoder import TreeDecoder
from src.models.projector import Projector

# === Metrics & Losses ===
from src.metrics.metrics_inference import compute_tree_result
from src.losses import contraclm_loss

# === Solvers ===
from src.solvers.gts.solver_lightning import MathSolver as SolverGTS
from src.solvers.gts_simpler_cl.solver_lightning import MathSolver as SolverSimpler
from src.solvers.gts_triplet_cl.solver_lightning import MathSolver as SolverTextual
from src.solvers.llm.solver_llm_lightning import MathSolver as SolverLLMSimpler
from src.solvers.llm.solver_llm_simpler_lightning import MathSolver as SolverLLM
from src.solvers.llm.solver_llm_contraclm_lightning import MathSolver as SolverLLMContraCLM

# === Constants ===
EMBEDDING_BERT = 768
EMBEDDING_SIZE = 128


def build_data_module(local_args):
    """Build data module based on the selected model."""
    if local_args.model == 'gts':
        return DataModuleGTS.MathDataModule(
            data_dir=f"data/{local_args.dataset}",
            train_file="train.jsonl",
            test_file="test.jsonl",
            tokenizer_path=local_args.pretrained_model,
            batch_size=local_args.batch_size,
            max_text_len=local_args.max_text_len,
        )

    if local_args.model == 'simpler':
        return DataModuleSimpler.MathDataModule(
            data_dir=f"data/{local_args.dataset}",
            train_file="train_simpler.jsonl",
            test_file="test.jsonl",
            tokenizer_path=local_args.pretrained_model,
            batch_size=local_args.batch_size,
            max_text_len=local_args.max_text_len,
        )

    if local_args.model == 'textual':
        return DataModuleTriplet.MathDataModule(
            data_dir=f"data/{local_args.dataset}",
            train_file=f"train_triplet_{local_args.similarity}.jsonl",
            test_file="test.jsonl",
            tokenizer_path=local_args.pretrained_model,
            batch_size=local_args.batch_size,
            max_text_len=local_args.max_text_len,
        )

    if local_args.model in ['llm', 'llm_simpler', 'llm_contraclm']:
        return DataModuleLLM.MathWordProblemDataModule(
            train_file=f"data/{local_args.dataset}/train_simpler.jsonl",
            dev_file=f"data/{local_args.dataset}/test.jsonl",
            tokenizer_checkpoint=local_args.pretrained_model,
            batch_size=local_args.micro_batch_size,
            max_length=local_args.max_length,
        )

    raise ValueError(f"Unknown model type: {local_args.model}")


def build_model(local_args, data_module):
    """Build model (Solver) based on model type."""
    tokenizer = data_module.tokenizer

    if local_args.model in ['gts', 'simpler', 'textual']:
        config = AutoConfig.from_pretrained(local_args.pretrained_model)
        pretrained_model = AutoModel.from_pretrained(local_args.pretrained_model)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        total_steps = len(data_module.train_dataloader()) * local_args.epoch

        encoder = Encoder(pretrained_model)
        decoder = TreeDecoder(
            config, len(data_module.op_tokens), len(data_module.constant_tokens), EMBEDDING_SIZE
        )

        if local_args.model == 'gts':
            if local_args.mode == 'demo':
                return SolverGTS.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    op_tokens=data_module.op_tokens,
                    constant_tokens=data_module.constant_tokens,
                    id_dict=data_module.id_dict,
                    total_steps=total_steps
                )
            return SolverGTS(
                local_args, encoder, decoder, tokenizer,
                data_module.op_tokens, data_module.constant_tokens, data_module.id_dict, total_steps
            )
        if local_args.model == 'simpler':
            projector = Projector(EMBEDDING_BERT, EMBEDDING_SIZE, len_subspace=3)
            if local_args.mode == 'demo':
                return SolverSimpler.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    encoder=encoder,
                    decoder=decoder,
                    projector=projector,
                    tokenizer=tokenizer,
                    op_tokens=data_module.op_tokens,
                    constant_tokens=data_module.constant_tokens,
                    id_dict=data_module.id_dict,
                    total_steps=total_steps
                )
            return SolverSimpler(
                local_args, encoder, decoder, projector, tokenizer,
                data_module.op_tokens, data_module.constant_tokens, data_module.id_dict, total_steps
            )
        if local_args.model == 'textual':
            if local_args.mode == 'demo':
                return SolverTextual.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    op_tokens=data_module.op_tokens,
                    constant_tokens=data_module.constant_tokens,
                    id_dict=data_module.id_dict,
                    total_steps=total_steps
                )
            return SolverTextual(
                local_args, encoder, decoder, tokenizer,
                data_module.op_tokens, data_module.constant_tokens, data_module.id_dict, total_steps
            )
    if local_args.model in ['llm', 'llm_simpler', 'llm_contraclm']:
        pretrained_model = AutoModelForCausalLM.from_pretrained(local_args.pretrained_model)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        local_args.accumulate_grad_batches = (
                local_args.batch_size // (local_args.micro_batch_size * len(local_args.devices)))

        if local_args.model == 'llm':
            if local_args.mode == 'demo':
                return SolverLLM.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    model=pretrained_model,
                    tokenizer=tokenizer,
                    train_dataset=data_module.train_dataset,
                    dev_dataset=data_module.dev_dataset
                )
            return SolverLLM(
                local_args, pretrained_model, tokenizer,
                data_module.train_dataset, data_module.dev_dataset
            )
        if local_args.model == 'llm_simpler':
            if local_args.mode == 'demo':
                return SolverLLMSimpler.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    model=pretrained_model,
                    tokenizer=tokenizer,
                    train_dataset=data_module.train_dataset,
                    dev_dataset=data_module.dev_dataset
                )
            return SolverLLMSimpler(
                local_args, pretrained_model, tokenizer,
                data_module.train_dataset, data_module.dev_dataset
            )
        if local_args.model == 'llm_contraclm':
            loss_func_seq = contraclm_loss.ContraCLMSeqLoss(
                pad_token_id=tokenizer.pad_token_id)
            if local_args.mode == 'demo':
                return SolverLLMContraCLM.load_from_checkpoint(
                    checkpoint_path=local_args.ckpt,
                    args=local_args,
                    model=pretrained_model,
                    tokenizer=tokenizer,
                    loss_func_seq=loss_func_seq,
                    train_dataset=data_module.train_dataset,
                    dev_dataset=data_module.dev_dataset
                )
            return SolverLLMContraCLM(
                local_args, pretrained_model, tokenizer, loss_func_seq,
                data_module.train_dataset, data_module.dev_dataset
            )


def preprocess_input(question):
    """
    Preprocess input question by replacing numbers with placeholders.

    Args:
        question (str): Raw input question text.

    Returns:
        tuple: Preprocessed question string and list of numbers extracted.
    """
    numbers = re.findall(r"\d+\.?\d*", question)
    number_map = {num: f"N_{i}" for i, num in enumerate(numbers)}

    for num, placeholder in number_map.items():
        question = question.replace(num, placeholder, 1)

    return question, [float(num) for num in numbers]


def inference_gts(model, number_tokens_ids, device, question):
    """
    Perform inference using GTS-based models.

    Args:
        model (LightningModule): Trained model.
        number_tokens_ids (list): List of special number token ids.
        device (torch.device): Inference device.
        question (str): Input question.

    Returns:
        tuple: Predicted prefix, infix, and final result.
    """
    text, nums = preprocess_input(question)

    text_ids = torch.tensor(
        model.tokenizer.encode("<O> " + text, max_length=256, truncation=True), dtype=torch.long
    )
    text_pads = torch.ones_like(text_ids, dtype=torch.float)
    num_ids = torch.tensor([i for i, s in enumerate(text_ids.tolist())
                            if s in number_tokens_ids], dtype=torch.long)
    num_pads = torch.ones_like(num_ids, dtype=torch.float)

    text_ids, text_pads, num_ids, num_pads = (
        text_ids.unsqueeze(0).to(device),
        text_pads.unsqueeze(0).to(device),
        num_ids.unsqueeze(0).to(device),
        num_pads.unsqueeze(0).to(device),
    )

    tree_out = model(text_ids, text_pads, num_ids, num_pads)
    pred_prefix, pred_infix, pred_result = compute_tree_result(tree_out, nums)

    return pred_prefix, pred_infix, pred_result


def inference_llm(local_args, model, device, question):
    """
    Perform inference using Decoder-only (LLM) models.

    Args:
        model (LightningModule): Trained model.
        device (torch.device): Inference device.
        question (str): Input question.

    Returns:
        tuple: Predicted prefix, infix, and final result.
    """
    text, nums = preprocess_input(question)

    input_text = f"Question: {text} Prefix: "
    inputs = model.tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=local_args.max_length
    )
    input_ids = inputs["input_ids"].to(device)
    outputs = model.model(input_ids=input_ids, output_hidden_states=True)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    pred = model.tokenizer.decode(predictions[0], skip_special_tokens=True)

    pred_prefix = pred.split("Prefix:")[-1].strip().split()
    pred_prefix, pred_infix, pred_result = compute_tree_result(pred_prefix, nums)

    return pred_prefix, pred_infix, pred_result


def run_demo_loop(local_args, model, data_module):
    """
    Launch interactive demo loop for user input.

    Args:
        local_args (Namespace): Parsed arguments.
        model (LightningModule): Trained model.
        data_module (LightningDataModule): Data module with tokenizer and datasets.
    """
    print("✅ Model loaded and ready for demo.")
    device = torch.device(f"cuda:{local_args.devices[0]}" if torch.cuda.is_available() else "cpu")
    if local_args.model in ['gts', 'simpler', 'textual']:
        model.solver.eval()
    else:
        model.model.eval()

    while True:
        user_input = input("\n🧮 Enter a math problem (or 'exit' to quit):\n")
        if user_input.lower() == "exit":
            break
        try:
            if local_args.model in ['gts', 'simpler', 'textual']:
                pred_prefix, pred_infix, pred_result = (
                    inference_gts(model, data_module.number_tokens_ids, device, user_input))
            else:
                pred_prefix, pred_infix, pred_result = (
                    inference_llm(local_args, model, device, user_input))

            print("📌 Infix:", pred_infix)
            print("📌 Prefix:", ' '.join(map(str, pred_prefix)))
            print("🔢 Result:", pred_result)
        except Exception:
            print("❌ Inference failed, check input or model state.")


def main(local_args):
    """
    Main function to run training, testing, or demo.

    Args:
        local_args (Namespace): Parsed command-line arguments.
    """
    pl.seed_everything(local_args.seed)
    data_module = build_data_module(local_args)
    data_module.setup()
    model = build_model(local_args, data_module)

    if local_args.mode == "demo":
        run_demo_loop(local_args, model, data_module)
    else:
        trainer_kwargs = model.set_trainer_kwargs()
        # Set limit_train_batches=0.01 for debug/test
        trainer = pl.Trainer(**trainer_kwargs)
        if local_args.mode == "test":
            trainer.test(model=model, datamodule=data_module, ckpt_path=local_args.ckpt)
        else:
            trainer.fit(model, datamodule=data_module)
            trainer.test(ckpt_path="best", datamodule=data_module)


def get_parser():
    """
    Argument parser for configuring training, testing, or demo.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser()
    # Basic settings
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['gts', 'simpler', 'textual', 'llm', 'llm_simpler', 'llm_contraclm'])
    parser.add_argument('--dataset', type=str, default='math23k', choices=['math23k', 'mathqa'])
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--similarity', type=str, default='tlwd', choices=['tlwd', 'ted'])
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--micro_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--log_step_ratio', default=0.001, type=float)
    parser.add_argument('--max_epochs', type=int, default=500)
    # Text & equation
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--max_equ_len', type=int, default=45)
    # Model paths
    parser.add_argument('--pretrained_model', type=str, default='hfl/chinese-bert-wwm-ext')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='experiments/tmp')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
