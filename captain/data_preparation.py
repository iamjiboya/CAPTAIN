import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
import numpy as np
import os

class SeqDataset(Dataset):
    """Dataset class for sequence data."""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_data_mouse(tokenized_train, mask_ratio, mask_value, pad_value, adata_protein, sort_seq_batch=False):
    """Prepare mouse data for training."""
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}")
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.ones_like(target_values_train).to(input_gene_ids_train.dtype),
    }
    return train_data_pt

def prepare_data_human(tokenized_train, mask_ratio, mask_value, pad_value, adata_protein, sort_seq_batch=False):
    """Prepare human data for training."""
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}")
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.zeros_like(target_values_train).to(input_gene_ids_train.dtype),
    }
    return train_data_pt

def prepare_dataloader(data_pt, batch_size, shuffle=False, intra_domain_shuffle=False, drop_last=False, num_workers=0):
    """Prepare a DataLoader for training."""
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    dataset = SeqDataset(data_pt)
    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
    return data_loader