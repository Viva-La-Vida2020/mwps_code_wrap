import argparse
import re
import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, BertTokenizer
import logging

from src.gts.models.solver import Solver
from src.gts.models.encoder import Encoder
from src.gts.models.tree_decoder import TreeDecoder
from src.metrics.metrics_inference import compute_tree_result
from data.data_module_gts import MathDataModule


class MathSolver(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a Math Word Problem solver.
    """

    def __init__(self, encoder, decoder, tokenizer, op_tokens, constant_tokens, id_dict):
        """
        Initializes the MathSolver.

        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            tokenizer (BertTokenizer): Tokenizer for text processing.
            op_tokens (list): List of operator tokens.
            constant_tokens (list): List of constant tokens.
            id_dict (dict): Dictionary mapping token IDs to tokens.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.op_tokens = op_tokens
        self.constant_tokens = constant_tokens
        self.id_dict = id_dict

        # Initialize the Solver (which encapsulates the encoder & decoder)
        self.solver = Solver(encoder, decoder)

    def forward(self, text_ids, text_pads, num_ids, num_pads):
        with torch.no_grad():
            tree_res = self.solver.evaluate_step(
                text_ids, text_pads, num_ids, num_pads,
                self.op_tokens, self.constant_tokens, max_length=45, beam_size=3
            )
            tree_out = [model.id_dict[x] for x in tree_res.out]

        return tree_out


def init_model(args):
    """
    Loads the trained model from the checkpoint.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        MathSolver: Loaded model.
        MathDataModule: Data module instance.
    """
    # Load DataModule
    data_module = MathDataModule(
        data_dir="data/mathqa",
        train_file="train.jsonl",
        test_file="test.jsonl",
        tokenizer_path=args.pretrained_model,
        batch_size=16,
        max_text_len=args.max_text_len,
    )
    data_module.setup()

    # Load Pretrained Model
    config = AutoConfig.from_pretrained(args.pretrained_model)
    tokenizer = data_module.tokenizer
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model, force_download=False)

    # Initialize Encoder & Decoder
    pretrained_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrained_model)
    decoder = TreeDecoder(config, len(data_module.op_tokens), len(data_module.constant_tokens), embedding_size=128)

    # Load trained model from checkpoint
    model = MathSolver.load_from_checkpoint(
        args.ckpt,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        op_tokens=data_module.op_tokens,
        constant_tokens=data_module.constant_tokens,
        id_dict=data_module.id_dict,
    )

    return model, data_module


def preprocess_input(question):
    """
    Preprocesses the user input by replacing numbers with placeholders.

    Args:
        question (str): User input question.

    Returns:
        str: Processed question with placeholders.
        list: List of extracted numbers.
    """
    numbers = re.findall(r"\d+\.?\d*", question)
    number_map = {num: f"N_{i}" for i, num in enumerate(numbers)}

    for num, placeholder in number_map.items():
        question = question.replace(num, placeholder, 1)

    return question, [float(num) for num in numbers]


def inference(model, number_tokens_ids, device, question):
    """
    Runs inference on a given math problem.

    Args:
        model (MathSolver): Trained model.
        number_tokens_ids (list): Token IDs for numbers.
        device (torch.device): Device to run inference on.
        question (str): User input question.

    Returns:
        list: Predicted prefix notation of the equation.
        list: Extracted numerical values from input.
    """
    text, nums = preprocess_input(question)

    text_ids = torch.tensor(
        model.tokenizer.encode("<O> " + text, max_length=256, truncation=True), dtype=torch.long
    )
    text_pads = torch.ones_like(text_ids, dtype=torch.float)
    num_ids = torch.tensor([i for i, s in enumerate(text_ids.tolist()) if s in number_tokens_ids], dtype=torch.long)
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


def get_parser():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='experiments/Test', type=str, help="Directory to save results.")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', default=1, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--pretrained_model', default='hfl/chinese-bert-wwm-ext', type=str, help="Pretrained model path.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--max_equ_len', default=45, type=int)
    parser.add_argument('--max_text_len', default=256, type=int)
    return parser


if __name__ == '__main__':
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    args = get_parser().parse_args()
    pl.seed_everything(args.seed)

    # Load model and data module
    model, data_module = init_model(args)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.solver.to(device)
    model.solver.eval()

    # Interactive Inference
    while True:
        user_input = input("\nEnter a math problem (type 'exit' to quit):\n")
        if user_input.lower() == "exit":
            break

        pred_prefix, pred_infix, pred_result = inference(model, data_module.number_tokens_ids, device, user_input)

        print("📌 Predicted Infix Expression:", pred_infix)
        print("📌 Predicted Prefix Expression:", pred_prefix)
        print("🔢 Predicted Answer:", pred_result)
