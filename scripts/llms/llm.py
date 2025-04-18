import torch
import argparse
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from data.data_module_llm import MathWordProblemDataModule
from src.utils.utils import save_json
from src.metrics.metrics import compute_prefix_tree_result
import os


class MathSolver(LightningModule):
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
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        loss, _, hidden_states = self.forward(input_ids, labels)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            scheduler.step()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss, logits, _ = self.forward(input_ids, labels)

        predictions = torch.argmax(logits, dim=-1)
        pred = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        pred_prefix = pred.split("Prefix:")[-1].strip().split()

        prefix = [op[0] for op in batch["prefix"]]  # Convert from [tuple(str, )] to [str, ]
        answer = batch["answer"]
        nums = [num[0] for num in batch["nums"]]

        val_ac, equ_ac = compute_prefix_tree_result(pred_prefix, prefix, answer, nums)

        self.val_correction.append(1 if val_ac else 0)
        self.equ_correction.append(1 if equ_ac else 0)

    def on_validation_epoch_end(self):
        total_val_acc = sum(self.val_correction) / len(self.val_correction)
        total_equ_acc = sum(self.equ_correction) / len(self.equ_correction)
        self.log("val_acc", total_val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("equ_acc", total_equ_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_correction.clear()
        self.equ_correction.clear()


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss, logits, _ = self.forward(input_ids, labels)

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
        save_json(self.test_results, os.path.join(self.save_path, './test_results.jsonl'))

        total_val_acc = sum(self.val_correction) / len(self.val_correction)
        total_equ_acc = sum(self.equ_correction) / len(self.equ_correction)
        self.log("val_acc", total_val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("equ_acc", total_equ_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_correction.clear()
        self.equ_correction.clear()

    def configure_optimizers(self):
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

        ret = dict(
            callbacks=callbacks,
            accelerator="gpu",
            # strategy="ddp",
            # strategy = None,
            default_root_dir=self.args.save_path,
            devices=self.args.devices,
            # accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=self.args.max_epochs,
            precision=self.args.precision,
        )

        ret.update(kwargs)
        return ret


def main(args):
    dm = MathWordProblemDataModule(
        train_file=args.train_file,
        dev_file=args.dev_file,
        tokenizer_checkpoint=args.pretrained_model,
        batch_size=args.micro_batch_size,
        max_length=args.max_length
    )
    dm.setup()
    accumulate_grad_batches = args.batch_size // (args.micro_batch_size * len(args.devices))

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    model.resize_token_embeddings(len(dm.tokenizer))

    lightning_model = MathSolver(
        model=model,
        tokenizer=dm.tokenizer,
        train_dataset=dm.train_dataset,
        dev_dataset=dm.dev_dataset,
        save_path=args.save_path,
        lr=args.learning_rate,
        accumulate_grad_batches=accumulate_grad_batches)

    trainer_kwargs = lightning_model.set_trainer_kwargs(
        default_root_dir=args.save_path,
        devices=args.devices,
        # accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=args.max_epochs,
        precision=args.precision,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.test:
        assert args.checkpoint is not None, f"args.checkpoint is required for test!"
        trainer.test(model=lightning_model, datamodule=dm)
    else:
        trainer.fit(lightning_model, datamodule=dm)
        model.save_pretrained(os.path.join(args.save_path, 'fine-tuned-model'))
        dm.tokenizer.save_pretrained(os.path.join(args.save_path, 'fine-tuned-model'))
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, default='../../data/math23k/train_simpler.jsonl', help='Path to the training data file')
    parser.add_argument('--dev_file', type=str, default='../../data/math23k/test.jsonl', help='Path to the development data file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=16, help='Micro Batch size for training')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--max_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--devices', type=int, nargs='+', default=[0, ], help='List of GPU devices to use')
    parser.add_argument('--precision', type=int, default=16, help='Precision for training, e.g., 16 for mixed precision')
    parser.add_argument('--save_path', type=str, default='experiments/math23k_gal1.3B_cl_Bs12', help='Directory to save')
    parser.add_argument('--test', action='store_true', default=False)
    # ['facebook/galactica-1.3b', 'facebook/galactica-125m', 'HuggingFaceTB/SmolLM-1.7B', 'HuggingFaceTB/SmolLM-135M']
    parser.add_argument('--pretrained_model', type=str, default='facebook/galactica-125m', help='Model checkpoint to use')
    args = parser.parse_args()

    main(args)