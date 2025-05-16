"""
Training script for GTS Math Word Problem Solver.
"""

import os
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from src.utils.utils import save_json
from src.metrics.metrics import compute_prefix_tree_result


class MathSolver(LightningModule):
    """
    PyTorch Lightning Module for training and evaluating the Math Word Problem solver.
    """

    def __init__(self, args, model, tokenizer, train_dataset, dev_dataset):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = self.model.config.hidden_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.save_path = args.save_path
        self.lr = args.lr
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.val_correction = []
        self.equ_correction = []
        self.test_results = []
        self.automatic_optimization = False

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        return outputs.loss, outputs.logits, outputs.hidden_states

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        loss, _, _ = self.forward(input_ids, labels)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            scheduler.step()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch):
        """
        Performs a single validation step.
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        _, logits, _ = self.forward(input_ids, labels)

        predictions = torch.argmax(logits, dim=-1)
        pred = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        pred_prefix = pred.split("Steps:")[-1].strip().split(" ; ")

        prefix = [op[0] for op in batch["prefix"]]  # Convert from [tuple(str, )] to [str, ]
        answer = batch["answer"]
        nums = [num[0] for num in batch["nums"]]

        val_ac, equ_ac = compute_prefix_tree_result(pred_prefix, prefix, answer, nums)

        self.val_correction.append(1 if val_ac else 0)
        self.equ_correction.append(1 if equ_ac else 0)

    def on_validation_epoch_end(self):
        """
        Computes dataset-wide validation accuracy at the end of an epoch.
        """
        total_val_acc = sum(self.val_correction) / len(self.val_correction)
        total_equ_acc = sum(self.equ_correction) / len(self.equ_correction)
        self.log("val_acc", total_val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("equ_acc", total_equ_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_correction.clear()
        self.equ_correction.clear()


    def test_step(self, batch):
        """
        Performs a single validation step.
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        _, logits, _ = self.forward(input_ids, labels)

        predictions = torch.argmax(logits, dim=-1)
        pred = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        pred_prefix = pred.split("Prefix:")[-1].strip().split()

        prefix = [op[0] for op in batch["prefix"]]  # Convert from [tuple(str, )] to [str, ]
        answer = batch["answer"]
        nums = [num[0] for num in batch["nums"]]

        val_correction, equ_correction = compute_prefix_tree_result(pred_prefix, prefix, answer, nums)
        result = {
            "prefix": prefix,
            "prediction": pred_prefix,
            "val_correction": val_correction,
            "equ_correction": equ_correction,
        }
        self.test_results.append(result)
        self.val_correction.append(1 if val_correction else 0)
        self.equ_correction.append(1 if equ_correction else 0)

    def on_test_epoch_end(self):
        """
        Computes dataset-wide validation accuracy at the end of an epoch.
        """
        save_json(self.test_results, os.path.join(self.save_path, './test_results.jsonl'))

        total_val_acc = sum(self.val_correction) / len(self.val_correction)
        total_equ_acc = sum(self.equ_correction) / len(self.equ_correction)
        self.log("val_acc", total_val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("equ_acc", total_equ_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_correction.clear()
        self.equ_correction.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler in Lightning
        """
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader()) * 500  # 500 epochs
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=16, shuffle=True)

    def set_trainer_kwargs(self, **kwargs):
        """
        Default kwargs used when initializing pl.Trainer
        """
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath=os.path.join(self.args.save_path, 'checkpoints'),
            filename="model-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            mode="max",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=50,
            verbose=True,
            mode="max"
        )
        lr_callback = LearningRateMonitor(logging_interval='step')
        callbacks += [checkpoint_callback, early_stop_callback, lr_callback]

        ret = {
            "callbacks": callbacks,
            "accelerator": "gpu",
            # "strategy": "ddp",
            # "strategy": None,
            "default_root_dir": self.args.save_path,
            "devices": self.args.devices,
            # "accumulate_grad_batches": accumulate_grad_batches,
            "max_epochs": self.args.max_epochs,
            "precision": self.args.precision,
        }
        ret.update(kwargs)
        return ret
