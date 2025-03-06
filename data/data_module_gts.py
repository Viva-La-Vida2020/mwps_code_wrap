import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import BertTokenizer, AutoConfig
from torch.nn.utils.rnn import pad_sequence


def load_data(filename):
    """ Load JSONL data """
    with open(filename, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class MathDataset(Dataset):
    """
    Custom Dataset for Math Word Problems (MWPs)
    """

    def __init__(self, mode, data_file, tokenizer, op_tokens, token_dict, number_tokens_ids, max_text_len=256):
        """
        Args:
            data_file (str): Path to the JSONL dataset file.
            tokenizer (BertTokenizer): Tokenizer for text processing.
            max_text_len (int): Maximum length of input text.
        """
        self.mode = mode
        self.data = load_data(data_file)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.op_tokens = op_tokens
        self.token_dict = token_dict
        self.number_tokens_ids = number_tokens_ids

    def _process_sample(self, sample):
        """ Tokenize and preprocess a single data sample """
        text_ids = torch.tensor(
            self.tokenizer.encode("<O> " + sample["text"], max_length=self.max_text_len, truncation=True),
            dtype=torch.long)
        equ_ids = torch.tensor([self.token_dict.get(x, len(self.op_tokens)) for x in sample["prefix"]],
                               dtype=torch.long)
        num_ids = torch.tensor([i for i, s in enumerate(text_ids.tolist()) if s in self.number_tokens_ids],
                               dtype=torch.long)

        if self.mode in ['train']:
            return {
                "text_ids": text_ids,
                "num_ids": num_ids,
                "equ_ids": equ_ids,
            }
        elif self.mode in ['test', 'val']:
            return {
                "text_ids": text_ids,
                "text_pads": torch.ones_like(text_ids, dtype=torch.float),
                "num_ids": num_ids,
                "num_pads": torch.ones_like(num_ids, dtype=torch.float),
                "equ_ids": equ_ids,
                "equ_pads": torch.ones_like(equ_ids, dtype=torch.float),
                "prefix": sample["prefix"],
                "nums": sample["nums"],
                "answer": sample["answer"],
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._process_sample(self.data[idx])


class MathDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Math Word Problems
    """

    def __init__(self, data_dir, train_file, test_file, tokenizer_path, batch_size=16, max_text_len=256):
        """
        Args:
            data_dir (str): Directory containing dataset.
            train_file (str): Training dataset file name.
            test_file (str): Test dataset file name.
            tokenizer_path (str): Path to pretrained tokenizer.
            batch_size (int): Batch size for DataLoader.
            max_text_len (int): Maximum sequence length.
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.tokenizer = None
        self.total_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.op_tokens, self.constant_tokens, self.number_tokens, self.token_dict, self.id_dict = None, None, None, None, None
        self.number_tokens_ids = None

    def setup(self, stage=None):
        """ Load tokenizer and initialize datasets """
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.total_data = load_data(f"{self.data_dir}/{self.train_file}") + load_data(f"{self.data_dir}/{self.test_file}")
        self.op_tokens, self.constant_tokens, self.number_tokens, self.token_dict, self.id_dict = self._build_token_dict()
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.number_tokens + ['<O>', '<Add>', '<Mul>']})
        self.number_tokens_ids = set([self.tokenizer.convert_tokens_to_ids(x) for x in self.number_tokens])

        self.train_dataset = MathDataset("train", f"{self.data_dir}/{self.train_file}", self.tokenizer, self.op_tokens, self.token_dict, self.number_tokens_ids, self.max_text_len)
        self.val_dataset = MathDataset("val", f"{self.data_dir}/{self.test_file}", self.tokenizer, self.op_tokens, self.token_dict, self.number_tokens_ids, self.max_text_len)
        self.test_dataset = MathDataset("test", f"{self.data_dir}/{self.test_file}", self.tokenizer, self.op_tokens, self.token_dict, self.number_tokens_ids, self.max_text_len)

    def _build_token_dict(self):
        """ Construct token dictionary from dataset """
        tokens = Counter()
        max_nums_len = 0

        for d in self.total_data:
            tokens += Counter(d["prefix"])
            max_nums_len = max(max_nums_len, len(d["nums"]))

        tokens = list(tokens)
        op_tokens = sorted([x for x in tokens if x[0].lower() not in {"c", "n"}])
        constant_tokens = sorted([x for x in tokens if x[0].lower() == "c"], key=lambda x: float(x[2:].replace("_", ".")))
        number_tokens = sorted([f"N_{i}" for i in range(max_nums_len)], key=lambda x: int(x[2:]))

        # Mapping token to ID
        token_list = op_tokens + constant_tokens + number_tokens
        token_dict = {x: i for i, x in enumerate(token_list)}
        id_dict = {x[1]: x[0] for x in token_dict.items()}

        return op_tokens, constant_tokens, number_tokens, token_dict, id_dict

    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle variable-length padding.
        """
        text_ids = pad_sequence([item["text_ids"] for item in batch], batch_first=True, padding_value=0)
        text_pads = (text_ids != self.tokenizer.pad_token_id).float()
        num_ids = pad_sequence([item["num_ids"] for item in batch], batch_first=True, padding_value=-1)
        num_pads = (num_ids != -1).float()
        equ_ids = pad_sequence([item["equ_ids"] for item in batch], batch_first=True, padding_value=-1)
        equ_pads = equ_ids != -1
        equ_ids[~equ_pads] = 0
        equ_pads = equ_pads.float()

        return {
            "text_ids": text_ids,
            "text_pads": text_pads,
            "num_ids": num_ids,
            "num_pads": num_pads,
            "equ_ids": equ_ids,
            "equ_pads": equ_pads
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                          collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0,)


if __name__ == '__main__':
    'Unit Test'
    # Initialize the DataModule
    data_module = MathDataModule(
        data_dir="math23k",  # Change this to your dataset directory
        train_file="train_cl.jsonl",  # Ensure this file exists in `data/`
        test_file="test.jsonl",  # Ensure this file exists in `data/`
        tokenizer_path="../pretrained_model/bert-base-chinese",  # Change if using another tokenizer
        batch_size=16,  # Small batch size for easy debugging
        max_text_len=256
    )

    # Setup data (mimics Lightning's internal workflow)
    data_module.setup()

    # Get the train dataloader
    train_loader = data_module.train_dataloader()

    # Iterate through a few batches
    print("🔍 Inspecting training batches...")

    for batch_idx, batch in enumerate(train_loader):
        print(f"\n🟢 Batch {batch_idx + 1}")

        print("📌 text_ids:", batch["text_ids"].shape)
        print("📌 text_pads:", batch["text_pads"].shape)
        print("📌 num_ids:", batch["num_ids"].shape)
        print("📌 equ_ids:", batch["equ_ids"].shape)

        # Print actual data
        print("\n📝 Sample Data:")
        print("text_ids:", batch["text_ids"][0].tolist())
        print("num_ids:", batch["num_ids"][0].tolist())
        print("equ_ids:", batch["equ_ids"][0].tolist())

        # Stop after inspecting 3 batches
        if batch_idx == 2:
            break
