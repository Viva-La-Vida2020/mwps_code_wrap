import os
import torch
import torch.nn.functional as F
import json
import argparse
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
# from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from models_LLM.mwps_dm import MathWordProblemDataModule
from models_LLM.utils import compute_prefix_result, save_json
from models_LLM.contrastive_losses import Subspace, TLWD_score_multiview, ted_score_multiview
from models_LLM.sup_con_loss import SupConLoss


class MWPsModule(LightningModule):
    def __init__(self, model, tokenizer, cl, similarity, train_dataset, dev_dataset, save_path, lr=2e-5,
                 accumulate_grad_batches=2):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cl = cl
        self.similarity = similarity
        self.hidden_dim = self.model.config.hidden_size
        self.subspace = Subspace(hidden_dim=self.hidden_dim, subspace_dim=128, len_subspace=3)
        self.criterion = SupConLoss()
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.save_path = save_path
        self.lr = lr
        self.accumulate_grad_batches = accumulate_grad_batches
        self.val_correction = []
        self.equ_correction = []
        self.test_results = []
        self.automatic_optimization = False

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        return outputs.loss, outputs.logits, outputs.hidden_states

    def cl_simpler(self, last_hidden_states, token_masks, prefixs, p_views, l_views):
        token_masks = token_masks.unsqueeze(-1)  # (N, L, 1)
        features = torch.sum(last_hidden_states * token_masks, dim=1) / torch.sum(token_masks, dim=1)  # [N, H]
        features_subspace = self.subspace(features)  # (N,3,128)
        if self.similarity == 'TLWD':
            scores = TLWD_score_multiview(prefixs, no_num=False)  # (N,N,3)
        elif self.similarity == 'TED':
            scores = ted_score_multiview(prefixs, p_views, l_views, no_num=False)

        scores = torch.from_numpy(scores).to(features.device)

        loss_c1 = self.criterion(torch.unsqueeze(features_subspace[:, 0, :], 1), mask=scores[:, 0, :])
        loss_c2 = self.criterion(torch.unsqueeze(features_subspace[:, 1, :], 1), mask=scores[:, 1, :])
        loss_c3 = self.criterion(torch.unsqueeze(features_subspace[:, 2, :], 1), mask=scores[:, 2, :])
        loss_cl = loss_c1 + loss_c2 + loss_c3
        return loss_cl

    def cl_simclr(self, last_hidden_states, token_masks, temperature=0.07):
        token_masks = token_masks.unsqueeze(-1)  # (N, L, 1)
        hidden_states_masked = last_hidden_states * token_masks
        sentence_embeddings = hidden_states_masked.sum(dim=1) / token_masks.sum(dim=1)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sim_matrix = torch.matmul(sentence_embeddings, sentence_embeddings.T)  # (N, N)

        batch_size = last_hidden_states.size(0)
        labels = torch.arange(batch_size).cuda()

        logits = sim_matrix / temperature
        loss = F.cross_entropy(logits, labels)

        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        token_masks = batch["token_mask"]  # (N, L)
        prefixs = batch['prefix']
        p_views = batch['p_view']
        l_views = batch['l_view']
        # Solver Loss
        loss_solver, _, hidden_states = self.forward(input_ids, labels)
        self.log("solver_loss", loss_solver, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        last_hidden_states = hidden_states[-1]  # Tensor:(N=16,L=256,H=576)
        if self.cl == 'SimplerCL':
            loss_cl = self.cl_simpler(last_hidden_states, token_masks, prefixs, p_views, l_views)
        elif self.cl == 'SimCLR':
            loss_cl = self.cl_simclr(last_hidden_states, token_masks)
        else:
            loss_cl = 0
        self.log("cl_loss", loss_cl, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        alpha = 0.05
        loss = loss_solver + alpha * loss_cl
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

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

        val_ac, equ_ac, _, _ = compute_prefix_result(pred_prefix, prefix, answer, nums)

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

        val_correction, equ_correction, _, _ = compute_prefix_result(pred_prefix, prefix, answer, nums)
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
            dirpath=os.path.join(args.save_path, 'checkpoints'),
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
            # strategy=None,
        )

        ret.update(kwargs)
        return ret


def main(args):
    dm = MathWordProblemDataModule(
        train_file=args.train_file,
        dev_file=args.dev_file,
        tokenizer_checkpoint=args.checkpoint,
        batch_size=args.micro_batch_size,
        max_length=args.max_length
    )
    dm.setup()
    accumulate_grad_batches = args.batch_size // (args.micro_batch_size * len(args.devices))

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.resize_token_embeddings(len(dm.tokenizer))

    lightning_model = MWPsModule(
        model=model,
        tokenizer=dm.tokenizer,
        cl=args.CL,
        similarity=args.similarity,
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

    parser.add_argument('--train_file', type=str, default='data/Math23k/Math23K_train.jsonl', help='Path to the training data file')
    parser.add_argument('--dev_file', type=str, default='data/Math23k/Math23K_test.jsonl', help='Path to the development data file')
    parser.add_argument('--CL', default='SimplerCL', type=str, choices=['SimplerCL', 'SimCLR', 'NoCL'])
    parser.add_argument('--similarity', default='TLWD', type=str, choices=['TLWD', 'TED'])
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Micro Batch size for training')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--max_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--devices', type=int, nargs='+', default=[0, ], help='List of GPU devices to use')
    parser.add_argument('--precision', type=int, default=16, help='Precision for training, e.g., 16 for mixed precision')
    parser.add_argument('--save_path', type=str, default='experiments/math23k_gal1.3B_cl_Bs4', help='Directory to save')
    parser.add_argument('--test', action='store_true', default=False)
    # ['facebook/galactica-1.3b', 'facebook/galactica-125m', 'HuggingFaceTB/SmolLM-1.7B', 'HuggingFaceTB/SmolLM-135M']
    parser.add_argument('--checkpoint', type=str, default='facebook/galactica-125m', help='Model checkpoint to use')
    args = parser.parse_args()

    main(args)
