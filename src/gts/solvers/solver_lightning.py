"""
Training script for GTS Math Word Problem Solver.
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from transformers import AdamW, get_linear_schedule_with_warmup

from src.gts.models.solver import Solver
from src.metrics.metrics import compute_prefix_tree_result
from src.utils.utils import save_json


class MathSolver(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating the Math Word Problem solver.
    """

    def __init__(self, args, encoder, decoder, tokenizer,
                 op_tokens, constant_tokens, id_dict, total_steps):
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
        self.test_results = []
        # self.automatic_optimization = False

    # pylint: disable=arguments-differ
    def forward(self, text_ids, text_pads, num_ids, num_pads):
        with torch.no_grad():
            tree_res = self.solver.evaluate_step(
                text_ids, text_pads, num_ids, num_pads,
                self.op_tokens, self.constant_tokens, max_length=45, beam_size=3
            )
            tree_out = [self.id_dict[x] for x in tree_res.out]

        return tree_out

    def training_step(self, batch):
        """
        Performs a single training step.

        Args:
            batch (dict): Input batch from DataLoader.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss, _ = self.solver.train_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            batch["equ_ids"], batch["equ_pads"], self.op_tokens, self.constant_tokens
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch):
        """
        Performs a single validation step.

        Args:
            batch (dict): Input batch from DataLoader.
        """
        tree_res = self.solver.evaluate_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            self.op_tokens, self.constant_tokens, self.hparams.max_equ_len, beam_size=3
        )

        tree_out, _ = tree_res.out, tree_res.score
        tree_out = [self.id_dict[x] for x in tree_out]

        # Individual sample correctness
        sample_val_correct, sample_equ_correct = compute_prefix_tree_result(
            tree_out,
            [x[0] for x in batch["prefix"]],
            float(batch["answer"]),
            [float(x[0]) for x in batch["nums"]]
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

    def test_step(self, batch):
        """
        Performs a single validation step.

        Args:
            batch (dict): Input batch from DataLoader.
        """
        tree_res = self.solver.evaluate_step(
            batch["text_ids"], batch["text_pads"], batch["num_ids"], batch["num_pads"],
            self.op_tokens, self.constant_tokens, self.hparams.max_equ_len, beam_size=3
        )

        tree_out, _ = tree_res.out, tree_res.score
        tree_out = [self.id_dict[x] for x in tree_out]

        # Individual sample correctness
        sample_val_correct, sample_equ_correct = compute_prefix_tree_result(
            tree_out,
            [x[0] for x in batch["prefix"]],
            float(batch["answer"]),
            [float(x[0]) for x in batch["nums"]]
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

        save_json(self.test_results, os.path.join(f"{self.hparams.save_path}/test_results.jsonl"))

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler in Lightning
        """
        optimizer = AdamW(self.solver.parameters(),
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
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

    def set_trainer_kwargs(self):
        """
        Default kwargs used when initializing pl.Trainer
        """
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.hparams.save_path)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.hparams.save_path,
                                          version=tb_logger.version)
        loggers = [tb_logger, csv_logger]
        log_every_n_steps = max(1, int(self.hparams.log_step_ratio * self.total_steps))

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath=os.path.join(tb_logger.save_dir,
                                 f"lightning_logs/version_{tb_logger.version}", "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.hparams.save_top_k,
            save_last=True,
            mode="max",
        )
        lr_callback = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_callback]

        return {
            "callbacks": callbacks,
            "logger": loggers,
            "log_every_n_steps": log_every_n_steps,
            "default_root_dir": self.hparams.save_path,
            "accelerator": "gpu",
            "devices": self.hparams.devices,
            "max_epochs": self.hparams.epoch,
            "check_val_every_n_epoch": self.hparams.check_val_every_n_epoch,
        }
