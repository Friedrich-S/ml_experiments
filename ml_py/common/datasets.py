from functools import partial
from typing import Dict, List, Tuple, Type
from datasets import load_dataset, Dataset
from .tokenizers import SimpleTokenizer, EncodeConfig
from torch.utils.data import DataLoader
import torch
from .model import ModelInput
from transformers import PreTrainedTokenizerBase


class SimpleCsvDataset:
    ds: Dataset

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def load(cls, path):
        return cls(load_dataset("csv", data_files=path)["train"])

    def get(self) -> Dataset:
        return self.ds

    def shuffled(self, seed):
        return SimpleCsvDataset(self.ds.shuffle(seed))

    def subset(self, count):
        return SimpleCsvDataset(self.ds.select(range(count)))

    def rename_columns(self, remap: Dict[str, str]):
        return SimpleCsvDataset(self.ds.rename_columns(remap))

    def map(self, fn, cache: bool = False):
        return SimpleCsvDataset(self.ds.map(fn, batched=True, keep_in_memory=not cache))

    def tokenize(
        self,
        tokenizer: SimpleTokenizer | PreTrainedTokenizerBase,
        enc_cfg: EncodeConfig,
        columns: List[str] | None = None,
        cache: bool = False,
    ):
        if isinstance(tokenizer, PreTrainedTokenizerBase):

            def map_fn(entry):
                for col in columns:
                    entry[col] = [
                        tokenizer(v, max_length=enc_cfg.max_length, padding=enc_cfg.pad)
                        for v in entry[col]
                    ]
                return entry

            return SimpleCsvDataset(
                self.ds.map(map_fn, batched=True, keep_in_memory=not cache)
            )

        if columns is None:

            def map_fn(entry):
                return {
                    k: [tokenizer.encode(v, enc_cfg) for v in v]
                    for k, v in entry.items()
                }

        else:

            def map_fn(entry):
                for col in columns:
                    entry[col] = [tokenizer.encode(v, enc_cfg) for v in entry[col]]
                return entry

        return SimpleCsvDataset(
            self.ds.map(map_fn, batched=True, keep_in_memory=not cache)
        )

    def jax(self):
        return SimpleCsvDataset(self.ds.with_format("jax"))

    def torch(self):
        return SimpleCsvDataset(self.ds.with_format("torch"))

    def train_test_split_loaders(
        self, seed: int, batch_size: int, test_perc: float, input_cls: Type[ModelInput]
    ) -> Tuple[DataLoader, DataLoader]:
        from jax import numpy as jnp

        def collate_fn(features):
            res = {}
            for k in features[0].keys():
                res[k] = jnp.stack([v[k] for v in features], axis=0)
            return input_cls.from_dict(res)

        ds = self.ds.train_test_split(test_perc)
        PartialDataLoader = partial(
            DataLoader, batch_size=batch_size, collate_fn=collate_fn
        )
        train_dl = PartialDataLoader(
            ds["train"],
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        test_dl = PartialDataLoader(
            ds["test"],
            shuffle=False,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )

        return train_dl, test_dl
