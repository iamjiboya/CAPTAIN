import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import Vocab
from torchtext._torchtext import VocabPybind
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, quantile_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from performer_pytorch import BLIP_Pretrain

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_pickle_file(file_path):
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def preprocss_rna(data, species, human_mouse_align, vocab_temp):
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

def preprocss_adt(data, species, adt_align_dict, adt_token_dict):
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

def our_step_preporcess(adata, adata_protein, species, human_mouse_align, vocab_temp, adt_align_dict, adt_token_dict):
    rna_data_pre = preprocss_rna(adata, species, human_mouse_align, vocab_temp)
    adt_data_pre = preprocss_adt(adata_protein, species, adt_align_dict, adt_token_dict)
    common_obs = rna_data_pre.obs_names.intersection(adt_data_pre.obs_names)
    rna_data_pre = rna_data_pre[common_obs]
    adt_data_pre = adt_data_pre[common_obs]
    return rna_data_pre, adt_data_pre

def prepare_data(tokenized, adata_protein, species, mask_ratio, mask_value, pad_value):
    masked_values = random_mask_value(
        tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids = tokenized["genes"]
    input_values = masked_values
    target_values = tokenized["values"]
    species_values = torch.zeros_like(target_values).to(input_gene_ids.dtype) if species == "human" else torch.ones_like(target_values).to(input_gene_ids.dtype)
    train_data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": target_values,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": species_values,
    }
    return train_data_pt

class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(data_pt, batch_size, shuffle=False, intra_domain_shuffle=False, drop_last=False, num_workers=0, distributed=False):
    dataset = SeqDataset(data_pt)
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader

class CombinedModel(nn.Module):
    def __init__(self, rna_model, adt_model):
        super(CombinedModel, self).__init__()
        self.rna_model = rna_model
        self.adt_model = adt_model

def train(model, loader, device, vocab, pad_token, optimizer, scaler, config):
    model.train()
    dist.barrier()
    total_loss = 0.0
    total_mse = 0.0
    total_cls = 0.0
    total_cce = 0.0
    start_time = time.time()
    num_batches = len(loader)
    criterion = masked_mse_loss
    criterion_quantile = quantile_loss
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            if isinstance(model, DDP):
                output_dict, transformer_out = model.module.rna_model(
                    input_gene_ids, input_values, species_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=config["CLS"], CCE=config["CCE"], MVC=config["MVC"], ECS=config["ECS"]
                )
                adt_embeddings, adt_to_out, adt_to_out_quantiles, _, labels_adt_data, adt_mask = model.module.adt_model(
                    adt_data, transformer_out, src_key_padding_mask, adt_values, output_atten=False
                )
            else:
                output_dict, transformer_out = model.rna_model(
                    input_gene_ids, input_values, species_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=config["CLS"], CCE=config["CCE"], MVC=config["MVC"], ECS=config["ECS"]
                )
                adt_embeddings, adt_to_out, adt_to_out_quantiles, _, labels_adt_data, adt_mask = model.adt_model(
                    adt_data, transformer_out, src_key_padding_mask, adt_values, output_atten=False
                )
            masked_positions = input_values.eq(-1)
            loss_mlm = 0.2 * criterion(output_dict["mlm_output"], target_values, masked_positions)
            loss_adt_mse = 0.8 * criterion(adt_to_out.squeeze(-1), labels_adt_data, adt_mask)
            loss_adt_quantile = 0.2 * criterion_quantile(adt_to_out_quantiles, labels_adt_data, adt_mask)
            loss = loss_mlm + loss_adt_mse + loss_adt_quantile
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_mse += loss_mlm.item()
        total_cls += loss_adt_mse.item()
        total_cce += loss_adt_quantile.item()
        if batch % 100 == 0 and batch > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / 100
            cur_loss = total_loss / 100
            cur_mse = total_mse / 100
            cur_cls = total_cls / 100
            cur_cce = total_cce / 100
            print(f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | cls {cur_cls:5.2f} | cce {cur_cce:5.2f}")
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            start_time = time.time()

def evaluate(model, loader, device, vocab, pad_token, save_dir):
    model.eval()
    predicted_adt_data = []
    true_adt_data = []
    with torch.no_grad():
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            species_values = batch_data["species_values"].to(device)
            adt_values = batch_data["adt_values"].to(device)
            adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
            if isinstance(model, DDP):
                output_dict, transformer_out = model.module.rna_model(
                    input_gene_ids, input_values, species_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=False, CCE=False, MVC=False, ECS=False
                )
                adt_embeddings, adt_to_out, adt_to_out_quantiles, _, labels_adt_data, _ = model.module.adt_model(
                    adt_data, transformer_out, src_key_padding_mask, adt_values, output_atten=False
                )
            else:
                output_dict, transformer_out = model.rna_model(
                    input_gene_ids, input_values, species_values, src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None, CLS=False, CCE=False, MVC=False, ECS=False
                )
                adt_embeddings, adt_to_out, adt_to_out_quantiles, _, labels_adt_data, _ = model.adt_model(
                    adt_data, transformer_out, src_key_padding_mask, adt_values, output_atten=False
                )
            predicted_adt_data.extend(adt_to_out.cpu().squeeze().tolist())
            true_adt_data.extend(labels_adt_data.cpu().squeeze().tolist())
    with open(os.path.join(save_dir, "predicted_adt_data.pickle"), 'wb') as file:
        pickle.dump(predicted_adt_data, file)
    with open(os.path.join(save_dir, "true_adt_data.pickle"), 'wb') as file:
        pickle.dump(true_adt_data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scRNA-seq and ADT data processing and model training/evaluation.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--rna_file", type=str, required=True)
    parser.add_argument("--adt_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="/home/jiboya/captain/results")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "eval" and args.load_model is None:
        parser.error("--load_model is required for evaluation mode")

    config = {
        "seed": args.seed,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_bins": 51,
        "MVC": False,
        "ECS": False,
        "CLS": False,
        "CCE": False,
        "INPUT_BATCH_LABELS": False,
        "DSBN": False,
        "amp": True,
        "dropout": 0.2,
        "layer_size": 128,
        "nlayers": 4,
        "nhead": 4,
        "fast_transformer": True,
        "pre_norm": False
    }

    set_seed(args.seed)

    if args.mode == "train":
        dist.init_process_group(backend='gloo')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_temp = read_json_file("/home/jiboya/captain/pretrain/vocab.json")
    human_mouse_align = read_pickle_file('/home/jiboya/captain/token_dict/human_mouse_align.pickle')
    adt_token_dict = read_pickle_file('/home/jiboya/captain/token_dict/adt_token_dict.pickle')
    adt_align_dict = read_pickle_file('/home/jiboya/captain/token_dict/adt_align_dict.pickle')

    adata = sc.read_h5ad(args.rna_file)
    adata_protein = sc.read_h5ad(args.adt_file)
    adata, adata_protein = our_step_preporcess(adata, adata_protein, args.species, human_mouse_align, vocab_temp, adt_align_dict, adt_token_dict)

    adata.var.set_index(adata.var.index, inplace=True)
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var["id_in_vocab"] = [1 if gene in vocab_temp else -1 for gene in adata.var["gene_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=config["n_bins"],
        result_binned_key="X_binned"
    )
    preprocessor(adata, batch_key=None)

    all_counts = adata.layers["X_binned"].A if sp.issparse(adata.layers["X_binned"]) else adata.layers["X_binned"]
    genes = adata.var["gene_name"].tolist()

    if args.load_model:
        model_dir = Path(args.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "our_PBMC_model.pt"
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
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
        vocab = Vocab(VocabPybind(genes + ["<pad>", "<cls>", "<eoc>"], None))
        vocab.set_default_index(vocab["<pad>"])
        embsize = config["layer_size"]
        nhead = config["nhead"]
        d_hid = config["layer_size"]
        nlayers = config["nlayers"]
        n_layers_cls = 3

    gene_ids = np.array(vocab(genes), dtype=int)
    tokenized_train = tokenize_and_pad_batch(
        all_counts, gene_ids, max_len=3001,
        vocab=vocab, pad_token="<pad>", pad_value=-2, append_cls=True, include_zero_gene=False
    )
    train_data_pt = prepare_data(tokenized_train, adata_protein, args.species, args.mask_ratio, -1, -2)
    train_loader = prepare_dataloader(
        train_data_pt, batch_size=args.batch_size, shuffle=False,
        intra_domain_shuffle=args.mode == "train", distributed=args.mode == "train"
    )

    model = TransformerModel(
        len(vocab), embsize, nhead, d_hid, nlayers, n_layers_cls, n_cls=1,
        vocab=vocab, dropout=config["dropout"], pad_token="<pad>", pad_value=-2,
        do_mvc=config["MVC"], do_dab=False,
        use_batch_labels=config["INPUT_BATCH_LABELS"], num_batch_labels=1,
        domain_spec_batchnorm=config["DSBN"], input_emb_style="continuous", n_input_bins=config["n_bins"],
        cell_emb_style="cls", mvc_decoder_style="inner product", ecs_threshold=0.0,
        explicit_zero_prob=False, use_fast_transformer=config["fast_transformer"], pre_norm=config["pre_norm"]
    )
    adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
    if args.load_model:
        try:
            rna_model_state_dict = {k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model')}
            model.load_state_dict(rna_model_state_dict)
            adt_model_state_dict = {k[len('module.adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.adt_model')}
            adt_model.load_state_dict(adt_model_state_dict)
            print(f"Loaded model from {model_file}")
        except:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model = CombinedModel(model, adt_model)
    model.to(device)
    if args.mode == "train":
        model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4 if config["amp"] else 1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    if args.mode == "train":
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, device, vocab, "<pad>", optimizer, scaler, config)
            torch.save(model.state_dict(), f"{args.save_dir}/{epoch}_model.pt")
    else:
        evaluate(model, train_loader, device, vocab, "<pad>", args.save_dir)