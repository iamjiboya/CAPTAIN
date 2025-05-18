from rna_model import TransformerModel, AdversarialDiscriminator
from protein_model import BLIP_Pretrain
from protein_model.loss import masked_mse_loss, quantile_loss, masked_relative_error, criterion_neg_log_bernoulli
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import pickle as pkl
import mudata as md
from mudata import MuData
import muon as mu
import os
import json
from tqdm import tqdm
import scipy.sparse as sp
from pathlib import Path
import shutil
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import random
from typing import List, Tuple, Dict, Union, Optional
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import scipy
from scipy.sparse import issparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_pickle_file(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def preprocss_rna(data, species):
    sc.pp.filter_genes(data, min_counts=10)
    sc.pp.filter_cells(data, min_counts=200)
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index
    rna_name = data.var.index.tolist()
    common_elements = set(rna_name) & set(vocab_temp.keys())
    print(f"Number of RNA genes in vocab: {len(common_elements)}")
    if len(common_elements) == 0:
        print("No matching genes found, exiting.")
        sys.exit()
    return data

def preprocss_adt(data, species):
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    sc.pp.scale(data)
    data.var = data.var.rename(index=adt_align_dict)
    data.var_names = data.var.index
    duplicated_genes = data.var_names.duplicated(keep='first')
    genes_to_keep = ~duplicated_genes
    data = data[:, genes_to_keep]
    gene_name = list(adt_token_dict.keys())
    adt_name = data.var.index.tolist()
    common_elements = set(adt_name) & set(gene_name)
    print(f"Number of ADT genes in vocab: {len(common_elements)}")
    if len(common_elements) == 0:
        print("No matching proteins found, exiting.")
        sys.exit()
    new_adata = ad.AnnData(np.zeros((data.shape[0], len(gene_name))), obs=data.obs.copy(), var=pd.DataFrame(index=gene_name))
    for gene in common_elements:
        if sp.issparse(data.X):
            try:
                new_adata.X[:, new_adata.var_names == gene] = data.X[:, data.var_names == gene].toarray()
            except IndexError:
                print(f"IndexError when processing {gene}")
                continue
        else:
            try:
                new_adata.X[:, new_adata.var_names == gene] = data.X[:, data.var_names == gene]
            except IndexError:
                print(f"IndexError when processing {gene}")
                continue
    return new_adata

def check_adata_x(adata):
    if sp.issparse(adata.X):
        non_zero_data = adata.X.data
        has_negative = (non_zero_data < 0).any()
        has_float = (non_zero_data != non_zero_data.astype(int)).any()
    else:
        has_negative = (adata.X < 0).any()
        has_float = (adata.X != adata.X.astype(int)).any()
    if has_negative or has_float:
        print("adata.X contains negative values or float values, which may cause problems in the downstream analysis.")
        sys.exit()

def our_step_preporcess(adata, adata_protein, species):
    check_adata_x(adata)
    check_adata_x(adata_protein)
    rna_data_pre = preprocss_rna(adata, species=species)
    adt_data_pre = preprocss_adt(adata_protein, species=species)
    common_obs = rna_data_pre.obs_names.intersection(adt_data_pre.obs_names)
    rna_data_pre = rna_data_pre[common_obs]
    adt_data_pre = adt_data_pre[common_obs]
    return rna_data_pre, adt_data_pre

def prepare_data_mouse(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value):
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.ones_like(target_values_train).to(input_gene_ids_train.dtype),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    return train_data_pt

def prepare_data_human(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value):
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.zeros_like(target_values_train).to(input_gene_ids_train.dtype),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    return train_data_pt

class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(data_pt, batch_size, shuffle=False, intra_domain_shuffle=False, drop_last=False, num_workers=0):
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    dataset = SeqDataset(data_pt)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

def train(model, loader):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
        celltype_labels = batch_data["celltype_labels"].to(device)
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict, transformer_out = model.rna_model(
                input_gene_ids,
                input_values,
                species_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )
            adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.adt_model(
                adt_data,
                transformer_out,
                src_key_padding_mask,
                adt_values,
                output_atten=False
            )
            loss = criterion_cls(celltype_mlp(adt_embeddings[:, -1, :]), celltype_labels)
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f}")
            total_loss = 0
            start_time = time.time()

def test(model, loader):
    model.eval()
    total_loss = 0.0
    predictions = []
    embs = []
    with torch.no_grad():
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            species_values = batch_data["species_values"].to(device)
            adt_values = batch_data["adt_values"].to(device)
            adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
            celltype_labels = batch_data["celltype_labels"].to(device)
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict, transformer_out = model.rna_model(
                    input_gene_ids,
                    input_values,
                    species_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                )
                adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.adt_model(
                    adt_data,
                    transformer_out,
                    src_key_padding_mask,
                    adt_values,
                    output_atten=False
                )
                output_values = celltype_mlp(adt_embeddings[:, -1, :])
                loss = criterion_cls(output_values, celltype_labels)
            total_loss += loss.item()
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
            embs.append(adt_embeddings[:, -1, :])
        print(total_loss)
        return np.concatenate(predictions, axis=0), embs

class Config:
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key, value)

class CombinedModel(nn.Module):
    def __init__(self, main_model, sub_model):
        super(CombinedModel, self).__init__()
        self.rna_model = main_model
        self.adt_model = sub_model

    def forward(self, x):
        pass

class Identity_Celltype(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super(Identity_Celltype, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scRNA-seq and ADT processing script.")
    parser.add_argument("--species", type=str, default="human", help="Species type: human or mouse")
    parser.add_argument("--train_gene_file", type=str, default="/home/jiboya/captain/cell_type_anno/dataset1/pbmc_gene_train.h5ad", help="Path to training gene data file")
    parser.add_argument("--train_protein_file", type=str, default="/home/jiboya/captain/cell_type_anno/dataset1/pbmc_protein_train.h5ad", help="Path to training protein data file")
    parser.add_argument("--test_gene_file", type=str, default="/home/jiboya/captain/cell_type_anno/dataset1/pbmc_gene_test.h5ad", help="Path to test gene data file")
    parser.add_argument("--test_protein_file", type=str, default="/home/jiboya/captain/cell_type_anno/dataset1/pbmc_protein_test.h5ad", help="Path to test protein data file")
    parser.add_argument("--save_dir", type=str, default="/home/jiboya/captain/cell_type_anno/dataset1/", help="Directory to save results")
    parser.add_argument("--load_model", type=str, default="/home/jiboya/captain_model", help="Path to load pre-trained model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=26, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    vocab_temp = read_json_file("/home/jiboya/captain/pretrain/vocab.json")
    human_mouse_align = read_pickle_file('/home/jiboya/captain/token_dict/human_mouse_align.pickle')
    adt_token_dict = read_pickle_file('/home/jiboya/captain/token_dict/adt_token_dict.pickle')
    adt_align_dict = read_pickle_file('/home/jiboya/captain/token_dict/adt_align_dict.pickle')

    hyperparameter_defaults = dict(
        seed=0,
        dataset_name="ms",
        do_train=True,
        load_model=args.load_model,
        mask_ratio=0.0,
        epochs=args.epochs,
        n_bins=51,
        MVC=False,
        ecs_thres=0.0,
        dab_weight=1.0,
        lr=args.lr,
        batch_size=args.batch_size,
        layer_size=512,
        nlayers=12,
        nhead=8,
        dropout=0.2,
        schedule_ratio=0.9,
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,
        include_zero_gene=False,
        freeze=False,
        DSBN=False,
        use_mod=True,
    )

    config = Config(hyperparameter_defaults)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    include_zero_gene = config.include_zero_gene
    max_seq_len = 3001
    n_bins = config.n_bins
    input_style = "binned"
    MLM = False
    CLS = False
    ADV = False
    CCE = False
    MVC = config.MVC
    ECS = config.ecs_thres > 0
    DAB = False
    INPUT_BATCH_LABELS = False
    input_emb_style = "continuous"
    cell_emb_style = "cls"
    mvc_decoder_style = "inner product"
    ecs_threshold = config.ecs_thres
    dab_weight = config.dab_weight
    explicit_zero_prob = MLM and include_zero_gene
    do_sample_in_train = False and explicit_zero_prob
    per_seq_batch_sample = False
    log_interval = 100
    save_eval_interval = config.save_eval_interval

    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")

    if config.load_model:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "CAPTAIN_Base.pt"
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        vocab = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab) if vocab else 0
    model = TransformerModel(
        ntokens,
        config.layer_size,
        config.nhead,
        config.layer_size,
        config.nlayers,
        nlayers_cls=3,
        n_cls=1 if CLS else 1,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=1,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=config.pre_norm,
    )
    if config.load_model:
        try:
            rna_model_state_dict = {
                k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model')
            }
            model.load_state_dict(rna_model_state_dict)
            print(f"Loading all model params from {model_file}")
        except:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
    adt_model_state_dict = {
        k[len('module.adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.adt_model')
    }
    adt_model.load_state_dict(adt_model_state_dict)

    model = CombinedModel(model, adt_model)
    model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    adata = sc.read_h5ad(args.train_gene_file)
    adata_protein = sc.read_h5ad(args.train_protein_file)
    adata, adata_protein = our_step_preporcess(adata, adata_protein, args.species)
    adata.var.set_index(adata.var.index, inplace=True)
    adata.var["gene_name"] = adata.var.index.tolist()
    adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    celltype_id_labels = adata.obs["celltype.l2"].astype("category").cat.codes.values
    num_types = len(np.unique(celltype_id_labels))
    adata.obs["celltype_id"] = celltype_id_labels
    celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=True,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=None)
    all_counts = adata.layers["X_binned"].A if sp.issparse(adata.layers["X_binned"]) else adata.layers["X_binned"]
    genes = adata.var["gene_name"].tolist()
    train_data = all_counts
    if config.load_model is None:
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    print(f"train set number of samples: {tokenized_train['genes'].shape[0]}, \n\t feature length: {tokenized_train['genes'].shape[1]}")
    if args.species == "human":
        train_data_pt = prepare_data_human(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value)
    elif args.species == "mouse":
        train_data_pt = prepare_data_mouse(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )

    adata = sc.read_h5ad(args.test_gene_file)
    adata_protein = sc.read_h5ad(args.test_protein_file)
    adata, adata_protein = our_step_preporcess(adata, adata_protein, args.species)
    adata.var.set_index(adata.var.index, inplace=True)
    adata.var["gene_name"] = adata.var.index.tolist()
    adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    celltype_id_labels = adata.obs["celltype.l2"].astype("category").cat.codes.values
    num_types = len(np.unique(celltype_id_labels))
    adata.obs["celltype_id"] = celltype_id_labels
    celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
    preprocessor(adata, batch_key=None)
    all_counts = adata.layers["X_binned"].A if sp.issparse(adata.layers["X_binned"]) else adata.layers["X_binned"]
    genes = adata.var["gene_name"].tolist()
    train_data = all_counts
    if config.load_model is None:
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    print(f"test set number of samples: {tokenized_train['genes'].shape[0]}, \n\t feature length: {tokenized_train['genes'].shape[1]}")
    if args.species == "human":
        train_data_pt = prepare_data_human(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value)
    elif args.species == "mouse":
        train_data_pt = prepare_data_mouse(tokenized_train, adata_protein, celltypes_labels, mask_ratio, mask_value, pad_value)
    test_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )

    celltype_mlp = Identity_Celltype(dropout=0., h_dim=256, out_dim=num_types).to(device)

    for epoch in range(1, config.epochs + 1):
        if config.do_train:
            train(model, loader=train_loader)
            predictions, embs = test(model, loader=test_loader)
            name = f"{save_dir}/{epoch}finetune_model.pt"
            torch.save(model.state_dict(), name)
            name = f"{save_dir}/{epoch}predictions.npy"
            np.save(name, predictions)
            name = f"{save_dir}/{epoch}embs.pt"
            torch.save([emb.cpu() for emb in embs], name)
            scheduler.step()