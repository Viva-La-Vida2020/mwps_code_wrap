import argparse
import os

from transformers import AutoModel, AutoConfig, BertTokenizer
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning import loggers as pl_loggers

from transformers import AdamW, get_linear_schedule_with_warmup
from src.gts_triplet_cl.models.solver import Solver  # Import the refactored Solver class
from src.gts_triplet_cl.models.encoder import Encoder
from src.gts_triplet_cl.models.tree_decoder import TreeDecoder
from src.metrics.metrics import compute_prefix_tree_result
from data.data_module_triplet import MathDataModule
from src.utils.utils import save_json


class MathSolver(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating the Math Word Problem solver.
    """

    def __init__(self, args, encoder, decoder, tokenizer, op_tokens, constant_tokens, id_dict, total_steps):
        """
        Initializes the MathSolver.

        Args:
            args (argparse.Namespace): Command-line arguments containing training configurations.
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            tokenizer (BertTokenizer): Tokenizer for text processing.
            op_tokens (list): List of operator tokens.
            constant_tokens (list): List of constant tokens.
            id_dict (dict): Dictionary mapping token IDs to tokens.
            total_steps (int): Total number of training steps for the scheduler.
        """
        super().__init__()

        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(args)

        self.tokenizer = tokenizer
        self.op_tokens = op_tokens
        self.constant_tokens = constant_tokens
        self.id_dict = id_dict
        self.total_steps = total_steps

        # Initialize the Solver (which encapsulates the encoder + decoder)
        self.solver = Solver(encoder, decoder)

        # Validation metrics accumulators
        self.correct_value_count = 0
        self.correct_equation_count = 0
        self.total_samples = 0
        # self.automatic_optimization = False

    def forward(self, text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads):
        """
        Forward pass for training.

        Args:
            text_ids (torch.Tensor): Input token IDs.
            text_pads (torch.Tensor): Input attention mask.
            num_ids (torch.Tensor): Number positions.
            num_pads (torch.Tensor): Number mask.
            equ_ids (torch.Tensor): Target equation token IDs.
            equ_pads (torch.Tensor): Target equation mask.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        loss, _ = self.solver.train_step(text_ids, text_pads, num_ids, num_pads, equ_ids, equ_pads, self.op_tokens,
                                         self.constant_tokens)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): Input batch from DataLoader.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss_solver, loss_cl, _ = self.solver.train_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            batch["pos_text_ids"], batch["pos_text_pads"], batch["pos_num_ids"], batch["pos_num_pads"],
            batch["neg_text_ids"], batch["neg_text_pads"], batch["neg_num_ids"], batch["neg_num_pads"],
            batch["equ_ids"], batch["equ_pads"], self.op_tokens, self.constant_tokens
        )
        loss = loss_solver + self.hparams.alpha * loss_cl

        self.log("loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("loss_solver", loss_solver, prog_bar=True, logger=True, on_epoch=True)
        self.log("loss_cl", loss_cl, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (dict): Input batch from DataLoader.
            batch_idx (int): Index of the batch.
        """
        tree_res = self.solver.evaluate_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            self.op_tokens, self.constant_tokens, self.hparams.max_equ_len, beam_size=3
        )

        tree_out, tree_score = tree_res.out, tree_res.score
        tree_out = [self.id_dict[x] for x in tree_out]

        # Individual sample correctness
        sample_val_correct, sample_equ_correct = compute_prefix_tree_result(
            tree_out, [x[0] for x in batch["prefix"]], float(batch["answer"]), [float(x[0]) for x in batch["nums"]]
        )

        # Accumulate for dataset-wide evaluation
        self.correct_value_count += 1 if sample_val_correct else 0
        self.correct_equation_count += 1 if sample_equ_correct else 0
        self.total_samples += 1

    def on_validation_epoch_end(self):
        """
        Computes dataset-wide validation accuracy at the end of an epoch.
        """
        value_accuracy = self.correct_value_count / self.total_samples
        equation_accuracy = self.correct_equation_count / self.total_samples

        # Reset accumulators for the next epoch
        self.correct_value_count = 0
        self.correct_equation_count = 0
        self.total_samples = 0

        # Log final validation accuracy for dataset
        self.log("val_acc", value_accuracy, prog_bar=True, logger=True)
        self.log("equ_acc", equation_accuracy, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (dict): Input batch from DataLoader.
            batch_idx (int): Index of the batch.
        """
        tree_res = self.solver.evaluate_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            self.op_tokens, self.constant_tokens, self.hparams.max_equ_len, beam_size=3
        )

        tree_out, tree_score = tree_res.out, tree_res.score
        tree_out = [self.id_dict[x] for x in tree_out]

        # Individual sample correctness
        sample_val_correct, sample_equ_correct = compute_prefix_tree_result(
            tree_out, [x[0] for x in batch["prefix"]], float(batch["answer"]), [float(x[0]) for x in batch["nums"]]
        )

        # Accumulate for dataset-wide evaluation
        self.correct_value_count += 1 if sample_val_correct else 0
        self.correct_equation_count += 1 if sample_equ_correct else 0
        self.total_samples += 1

        result = {
            "prefix": [x[0] for x in batch["prefix"]],
            "prediction": tree_out,
            "val_correction": sample_val_correct,
            "equ_correction": sample_equ_correct,
        }
        self.test_results.append(result)

    def on_test_epoch_end(self):
        """
        Computes dataset-wide validation accuracy at the end of an epoch.
        """
        value_accuracy = self.correct_value_count / self.total_samples
        equation_accuracy = self.correct_equation_count / self.total_samples

        # Reset accumulators for the next epoch
        self.correct_value_count = 0
        self.correct_equation_count = 0
        self.total_samples = 0

        # Log final validation accuracy for dataset
        self.log("val_acc", value_accuracy, prog_bar=True, logger=True)
        self.log("equ_acc", equation_accuracy, prog_bar=True, logger=True)

        save_json(self.test_results, os.path.join(self.hparams.pretrained_model, f"test/{self.hparams.ckpt}/test_results.jsonl"))

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler in Lightning
        """
        optimizer = AdamW(self.solver.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.total_steps * self.hparams.warmup_ratio,
                num_training_steps=self.total_steps,
            ),
            'interval': 'step',  # The scheduler will update every step
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def set_trainer_kwargs(self, **kwargs):
        """
        Default kwargs used when initializing pl.Trainer
        """
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.hparams.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.hparams.save_dir, version=tb_logger.version)
        loggers = [tb_logger, csv_logger]
        log_every_n_steps = max(1, int(self.hparams.log_step_ratio * self.total_steps))

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath=os.path.join(tb_logger.save_dir, f"lightning_logs/version_{tb_logger.version}", "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.hparams.save_top_k,
            save_last=True,
            mode="max",
        )
        lr_callback = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_callback]

        ret = dict(
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=log_every_n_steps,
            default_root_dir=self.hparams.save_dir,
            accelerator="gpu",
            devices=self.hparams.devices,
            max_epochs=self.hparams.epoch,
            check_val_every_n_epoch=self.hparams.check_val_every_n_epoch,
            # strategy="ddp",
            # limit_train_batches=0.1,  # Only use 10% of the training data
            # limit_val_batches=0.01,  # Only use 10% of validation data
        )

        return ret


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='experiments/Test', type=str)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, ], help='List of GPU devices to use')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--max_equ_len', default=45, type=int)
    parser.add_argument('--max_text_len', default=256, type=int)
    parser.add_argument('--save_top_k', default=1, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--log_step_ratio', default=0.001, type=float)
    parser.add_argument('--dataset', default='math23k', type=str, choices=['math23k', 'mathqa'])
    parser.add_argument('--pretrained_model', default='hfl/chinese-bert-wwm-ext', type=str,
                        choices=['hfl/chinese-bert-wwm-ext', 'hfl/chinese-roberta-wwm-ext', 'bert-base-uncased'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    pl.seed_everything(args.seed)

    # Load DataModule
    data_path = f"data/{args.dataset}"
    data_module = MathDataModule(
        data_dir=data_path,
        train_file="train_triplet_tlwd.jsonl",
        test_file="test.jsonl",
        tokenizer_path=args.pretrained_model,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
    )

    data_module.setup()

    # Load Pretrained Model
    config = AutoConfig.from_pretrained(args.pretrained_model)
    tokenizer = data_module.tokenizer
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model)

    # Initialize Encoder & Decoder
    pretrained_model.resize_token_embeddings(len(tokenizer))
    encoder = Encoder(pretrained_model)
    decoder = TreeDecoder(config, len(data_module.op_tokens), len(data_module.constant_tokens), embedding_size=128)

    # Calculate total training steps for scheduler
    total_steps = len(data_module.train_dataloader()) * args.epoch

    # Initialize MathSolver Model
    model = MathSolver(
        args=args,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        op_tokens=data_module.op_tokens,
        constant_tokens=data_module.constant_tokens,
        id_dict=data_module.id_dict,
        total_steps=total_steps,
    )

    # Define Trainer
    trainer_kwargs = model.set_trainer_kwargs()
    trainer = pl.Trainer(**trainer_kwargs)

    if args.test:
        trainer.test(model, datamodule=data_module, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, datamodule=data_module)
        trainer.test(ckpt_path="best", datamodule=data_module)


