import torch
from torch import nn


def gather_vectors(encoded: torch.Tensor, num_ids: torch.Tensor) -> torch.Tensor:
    """
    Gather specific positions from encoded sequences based on num_ids.

    Args:
        encoded (torch.Tensor): Shape (batch_size, seq_len, hidden_size) - Encoded token representations.
        num_ids (torch.Tensor): Shape (batch_size, max_num_count) - Indices of numbers in text.

    Returns:
        torch.Tensor: Shape (batch_size, max_num_count, hidden_size) - Gathered representations.
    """
    batch_size, _, hidden_size = encoded.shape
    max_num_count = num_ids.shape[1]

    # Initialize tensor with zeros
    gathered = torch.zeros(batch_size, max_num_count, hidden_size, dtype=encoded.dtype, device=encoded.device)

    # Mask positions where num_ids == -1 (invalid indices)
    valid_mask = num_ids >= 0

    # Only gather valid indices
    if valid_mask.any():
        gathered[valid_mask] = encoded[torch.arange(batch_size).unsqueeze(-1), num_ids.clamp(min=0)][valid_mask]

    return gathered


class Encoder(nn.Module):
    """
    Encoder model wrapping a pretrained Transformer model.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the Encoder module.

        Args:
            model (nn.Module): Pretrained Transformer-based model.
        """
        super().__init__()
        self.model = model

    def forward(self, text_ids: torch.Tensor, text_pads: torch.Tensor, num_ids: torch.Tensor, num_pads: torch.Tensor):
        """
        Forward pass for encoding text sequences and extracting number representations.

        Args:
            text_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            text_pads (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            num_ids (torch.Tensor): Indices of numbers in the text of shape (batch_size, max_num_count).
            num_pads (torch.Tensor): Mask indicating valid numbers in num_ids (batch_size, max_num_count).

        Returns:
            dict: Dictionary containing:
                - 'text': Encoded text representations (batch_size, seq_len, hidden_size).
                - 'text_pads': Attention mask (batch_size, seq_len).
                - 'num': Extracted number embeddings (batch_size, max_num_count, hidden_size).
                - 'num_pads': Number mask (batch_size, max_num_count).
        """
        outputs = self.model(input_ids=text_ids, attention_mask=text_pads)
        encoded = outputs.last_hidden_state  # Equivalent to outputs[0] (better readability)
        num_representations = gather_vectors(encoded, num_ids)

        return {
            "text": encoded,
            "text_pads": text_pads,
            "num": num_representations,
            "num_pads": num_pads,
        }

    def save_pretrained(self, save_directory: str):
        """
        Saves the pretrained model.

        Args:
            save_directory (str): Path to save the model weights.
        """
        self.model.save_pretrained(save_directory)
