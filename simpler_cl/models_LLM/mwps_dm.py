import torch
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class MWPTrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        prefix = " ".join(item["prefix"])
        input_text = f"Question: {text} Prefix: "
        target_text = f"Question: {text} Prefix: {prefix}"
        input_ids = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length)[
            "input_ids"]
        label_ids = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length)[
            "input_ids"]
        token_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label_ids),
            "token_mask": torch.tensor(token_mask, dtype=torch.bool),
            "prefix": item["prefix"],
            "p_view": item["root_nodes"],
            "l_view": item["longest_view"],
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    token_masks = torch.stack([item["token_mask"] for item in batch])

    prefixes = [item["prefix"] for item in batch]
    p_views = [item["p_view"] for item in batch]
    l_views = [item["l_view"] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "token_mask": token_masks,
        "prefix": prefixes,
        "p_view": p_views,
        "l_view": l_views,
    }


class MWPTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        prefix = " ".join(item["prefix"])
        input_text = f"Question: {text} Prefix: "  # e.g."Question: I have N_0 apples and N_1 bananas. How many fruit I have in total? Prefix:"
        target_text = f"Question: {text} Prefix: {prefix}"  # e.g."Question: I have N_0 apples and N_1 bananas. How many fruit I have in total? Prefix: + N_0 N_1"
        input_ids = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length)[
            "input_ids"]
        label_ids = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length)[
            "input_ids"]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label_ids),
            "prefix": item["prefix"],
            "answer": item["answer"],
            "nums": item["nums"]
        }


class MathWordProblemDataModule(LightningDataModule):
    def __init__(self, train_file, dev_file, tokenizer_checkpoint, batch_size=16, max_length=256):
        super().__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        train_data = self.load_json(self.train_file)
        dev_data = self.load_json(self.dev_file)

        self.train_dataset = MWPTrainDataset(train_data, self.tokenizer, max_length=self.max_length)
        self.dev_dataset = MWPTestDataset(dev_data, self.tokenizer, max_length=self.max_length)

    def load_json(self, filename):
        data = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          collate_fn=custom_collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=1, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=1, num_workers=8)