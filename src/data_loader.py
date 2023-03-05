"""
This is data loader class file
"""
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from src.encodings import PyTorchDataModule


class T5DataModule(pl.LightningDataModule):
    """ T5 data class """

    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int = 4,
                 source_max_token_length: int = 512,
                 target_max_token_length: int = 512,
                 num_workers: int = 2
                 ):
        """ initiates a T5 Data Module """
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length
        self.num_workers = num_workers

    def setup(self):
        """ getting train_dataset, test_dataset """

        self.train_dataset = PyTorchDataModule(self.train_df, self.tokenizer, self.source_max_token_length, self.target_max_token_length)
        self.val_dataset = PyTorchDataModule(self.val_df, self.tokenizer, self.source_max_token_length, self.target_max_token_length)

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
