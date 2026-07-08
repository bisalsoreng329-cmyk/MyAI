import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Creates (input, target) pairs for next-token prediction.
    """

    def __init__(self, text, tokenizer, seq_length=64):
        self.seq_length = seq_length
        self.tokens = tokenizer.encode(text)

        if len(self.tokens) <= seq_length:
            raise ValueError("Text is too short for the selected sequence length.")

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index):
        x = self.tokens[index:index+self.seq_length]
        y = self.tokens[index+1:index+self.seq_length+1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
