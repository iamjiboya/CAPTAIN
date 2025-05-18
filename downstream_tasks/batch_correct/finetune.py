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
        data = pickle.load(fp) # Changed from pkl to pickle
    return data

def preprocss_rna(data: ad.AnnData, species: str, human_mouse_align: Dict, vocab_temp: Dict) -> ad.AnnData:
    if species == "mouse":

        data.var.index = data.var.index.astype(str)

        new_var_names = []
        original_var_names = []
        for gene in data.var_names:
            if gene in human_mouse_align:
                new_var_names.append(human_mouse_align[gene])
                original_var_names.append(gene)
        
        # Filter AnnData object by genes that were mapped
        data = data[:, original_var_names].copy()
        # Rename var_names to human orthologs
        data.var_names = new_var_names
        data.var.index = new_var_names


    rna_name = data.var.index.tolist()
    common_elements = set(rna_name) & set(vocab_temp.keys())
    print(f"Number of RNA genes in vocab: {len(common_elements)}")
    if len(common_elements) == 0:
        print("No matching RNA genes found in vocab, exiting.")
        sys.exit()
    # Filter data to keep only common genes
    data = data[:, list(common_elements)].copy()
    return data

def our_step_preporcess(adata: ad.AnnData, species: str, human_mouse_align: Dict, vocab_temp: Dict) -> ad.AnnData:
    rna_data_pre = preprocss_rna(adata, species, human_mouse_align, vocab_temp)

    return rna_data_pre

def prepare_data(tokenized: Dict[str, np.ndarray], species: str, 
                 mask_ratio: float, mask_value: int, pad_value: int, 
                 batch_ids: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
    masked_values = random_mask_value(
        tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids = torch.from_numpy(tokenized["genes"])
    input_values = torch.from_numpy(masked_values)
    target_values = torch.from_numpy(tokenized["values"])
    

    species_id = 0 if species == "human" else 1
    species_values_tensor = torch.full((input_gene_ids.shape[0],), species_id, dtype=input_gene_ids.dtype)

    train_data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": target_values,
        "species_values": species_values_tensor, # Changed to be per sample
    }
    if batch_ids is not None:
        train_data_pt["batch_labels"] = torch.from_numpy(batch_ids).long()
    return train_data_pt

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self) -> int:
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, shuffle: bool = False, 
                       drop_last: bool = False, num_workers: int = 0, 
                       distributed: bool = False) -> DataLoader:
    dataset = SeqDataset(data_pt)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False, # Shuffle is handled by sampler if distributed
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader

def train(model: nn.Module, loader: DataLoader, device: torch.device, 
          vocab: Vocab, pad_token: str, optimizer: torch.optim.Optimizer, 
          scaler: torch.cuda.amp.GradScaler, config: Dict):
    model.train()
    if dist.is_initialized():
        dist.barrier()
        
    total_loss_epoch = 0.0
    total_mlm_loss_epoch = 0.0 # For tracking MLM component if needed

    log_loss_total = 0.0
    log_loss_mlm = 0.0
    
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device) # Still needed if MLM is explicitly calculated or for inspection
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device) if "batch_labels" in batch_data else None

        with torch.cuda.amp.autocast(enabled=config.get("amp", True)):
            # Determine if DDP is used to call the model correctly
            # The model itself is TransformerModel now
            if isinstance(model, DDP):
                output_dict, _ = model.module( # _ was transformer_out, not needed further
                    input_gene_ids, 
                    input_values, 
                    species_values, # Pass species_values
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.get("INPUT_BATCH_LABELS") or config.get("DSBN") else None,
                    CLS=config.get("CLS", False), 
                    CCE=config.get("CCE", False), 
                    MVC=config.get("MVC", True), # Defaulting MVC to True as per original config
                    ECS=config.get("ECS", False)
                )
            else:
                output_dict, _ = model( # _ was transformer_out, not needed further
                    input_gene_ids, 
                    input_values, 
                    species_values, # Pass species_values
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.get("INPUT_BATCH_LABELS") or config.get("DSBN") else None,
                    CLS=config.get("CLS", False), 
                    CCE=config.get("CCE", False), 
                    MVC=config.get("MVC", True), 
                    ECS=config.get("ECS", False)
                )
            
            loss = output_dict["loss"] # Use the loss from the model
            mlm_loss_from_model = output_dict.get("loss_mlm", torch.tensor(0.0)).item() # For logging

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_epoch += loss.item()
        total_mlm_loss_epoch += mlm_loss_from_model
        log_loss_total += loss.item()
        log_loss_mlm += mlm_loss_from_model


        if (batch_idx + 1) % 100 == 0 and batch_idx > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / 100 # Corrected: 100 batches
            cur_loss = log_loss_total / 100
            cur_mlm_loss = log_loss_mlm / 100
            
            print(f"| epoch {epoch_num:3d} | {batch_idx+1:5d}/{len(loader):5d} batches | "
                  f"lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | "
                  f"loss {cur_loss:5.4f} | mlm_loss {cur_mlm_loss:5.4f}")
            log_loss_total = 0.0
            log_loss_mlm = 0.0
            start_time = time.time()
            
    return total_loss_epoch / len(loader), total_mlm_loss_epoch / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
             vocab: Vocab, pad_token: str, save_dir: str, config: Dict): # Added config
    model.eval()
    # Example: Collect RNA embeddings (CLS token output)
    all_rna_embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            species_values = batch_data["species_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device) if "batch_labels" in batch_data else None

            # Determine if DDP is used
            if isinstance(model, DDP):
                output_dict, out_others = model.module(
                    input_gene_ids, 
                    input_values, 
                    species_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.get("INPUT_BATCH_LABELS") or config.get("DSBN") else None,
                    CLS=True, # Assuming CLS token is desired for embedding
                    CCE=config.get("CCE", False), 
                    MVC=config.get("MVC", True), 
                    ECS=config.get("ECS", False)
                )
            else:
                output_dict, out_others = model(
                    input_gene_ids, 
                    input_values, 
                    species_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.get("INPUT_BATCH_LABELS") or config.get("DSBN") else None,
                    CLS=True, 
                    CCE=config.get("CCE", False), 
                    MVC=config.get("MVC", True), 
                    ECS=config.get("ECS", False)
                )

            if config.get("CLS", True) and "cls_output" in out_others: # CLS should be True in model for this
                 all_rna_embeddings.append(out_others["cls_output"].cpu())

    if all_rna_embeddings:
        rna_embeddings_tensor = torch.cat(all_rna_embeddings, dim=0)
        print(f"Collected RNA embeddings of shape: {rna_embeddings_tensor.shape}")
        # Save RNA embeddings
        save_path = Path(save_dir) / "rna_embeddings.pt"
        torch.save(rna_embeddings_tensor, save_path)
        print(f"Saved RNA embeddings to {save_path}")
    else:
        print("No RNA embeddings collected (possibly CLS was False or not in output).")


# Global variable for epoch number, to be used in logging within train function
epoch_num = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scRNA-seq data processing and model training/evaluation.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="Mode to run: train or eval.")
    parser.add_argument("--species", type=str, default="human", choices=["human", "mouse"], help="Species of the dataset.")
    parser.add_argument("--rna_file", type=str, required=True, help="Path to the RNA data file (e.g., .h5ad).")
    # --adt_file removed
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results and models.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pre-trained model directory or specific model file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="Ratio of values to mask for MLM (if model uses it). Original script had 0.0, scGPT typically uses 0.15 for pretrain")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.") # Original used 1e-3, scGPT often uses 1e-4
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training. Default -1 means not using DDP explicitly here, but will be set by launch utility.")

    args = parser.parse_args()

    if args.mode == "eval" and args.load_model is None:
        parser.error("--load_model is required for evaluation mode")

    # Create save_dir if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Configuration dictionary
    # These are scGPT default like parameters, adjust as needed
    config = {
        "seed": args.seed,
        "mask_ratio": args.mask_ratio, # Used in prepare_data
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        
        "embsize": 512, # Transformer embedding size
        "d_hid": 512,  # Dimension of the feedforward network model in nn.TransformerEncoder
        "nlayers": 12, # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "nhead": 8,    # Number of heads in nn.MultiheadAttention
        "dropout": 0.1, # Dropout rate, scGPT uses 0.2, but 0.1 is common too
        "n_bins": 51,  # Number of bins for value embedding
        
        "MVC": True,   # Masked Value Prediction for genes, True if model supports it
        "ECS": False,  # Expression Conditioned Similarity, False
        "CLS": True,   # Cell type classification (or other CLS token tasks), True to get CLS embedding
        "CCE": False,  # Contrastive Cell Embedding, False
        
        "INPUT_BATCH_LABELS": True if args.mode == "train" else False, # Whether to use batch labels if available
        "DSBN": False, # Domain-Specific Batch Normalization
        
        "amp": True, # Automatic Mixed Precision
        "fast_transformer": True, # Use flash attention if available
        "pre_norm": False # Normalization before or after multi-head attention
    }
    
    # Save config to file
    with open(Path(args.save_dir) / "args_config.json", "w") as f:
        json.dump(vars(args), f, indent=4) # Save command line args
    with open(Path(args.save_dir) / "model_config.json", "w") as f:
        json.dump(config, f, indent=4) # Save model config

    set_seed(config["seed"])

    # Distributed training setup
    is_distributed = args.local_rank != -1
    if is_distributed:
        if not dist.is_initialized(): # Check if already initialized by launch utility
          dist.init_process_group(backend='nccl') # 'gloo' was in original, 'nccl' is common for GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f"Running DDP on rank {args.local_rank}, device {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device {device}")


    base_path = Path("/home/jiboya/captain/") # Example base path from original script
    vocab_file_path = base_path / "pretrain/vocab.json"
    human_mouse_align_file_path = base_path / "token_dict/human_mouse_align.pickle"

    if not vocab_file_path.exists():
        sys.exit(f"RNA vocab file not found at {vocab_file_path}")
    if args.species == "mouse" and not human_mouse_align_file_path.exists():
        sys.exit(f"Human-mouse alignment file not found at {human_mouse_align_file_path}")

    vocab_temp = read_json_file(vocab_file_path)
    human_mouse_align = {}
    if args.species == "mouse":
        human_mouse_align = read_pickle_file(human_mouse_align_file_path)

    adata = sc.read_h5ad(args.rna_file)
    adata = our_step_preporcess(adata, args.species, human_mouse_align, vocab_temp)

    # Further AnnData processing specific to scGPT
    adata.var["gene_name"] = adata.var.index.astype(str).tolist()

    batch_ids = None
    if args.mode == "train" and "batch_id" in adata.obs.columns:
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        adata.obs["batch_id_encoded"] = le.fit_transform(adata.obs['batch_id'].astype(str).values)
        batch_ids = np.array(adata.obs["batch_id_encoded"].tolist())
        config["num_batch_labels"] = len(le.classes_)
        print(f"Using {len(le.classes_)} batch labels for training.")
    else:
        config["INPUT_BATCH_LABELS"] = False # Ensure it's False if no batch_id
        config["num_batch_labels"] = 1 # Default for model if not used

    # scGPT Preprocessor
    preprocessor = Preprocessor(
        use_key="X",  # Layer to use for processing
        filter_gene_by_counts=False,  # Genes already filtered
        filter_cell_by_counts=False, # Cells not filtered here
        normalize_total=1e4, # Standard scRNA-seq normalization
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False, # HVG selection turned off, assuming all genes from vocab are used or pre-filtered
        # subset_hvg=1200 if args.mode == "train" else False, # Original had this
        # hvg_flavor="seurat_v3",
        binning=config["n_bins"], # Binning for value embedding
        result_binned_key="X_binned"
    )
    preprocessor(adata, batch_key="batch_id_encoded" if batch_ids is not None else None) # Pass batch_key if DSBN or similar is used

    # Use binned data for tokenization as per scGPT
    if "X_binned" not in adata.layers:
        sys.exit("Error: 'X_binned' layer not found after preprocessing. Check Preprocessor steps.")
    
    # Ensure data is not all zero after binning for any cell
    if isinstance(adata.layers["X_binned"], np.ndarray):
        if np.all(adata.layers["X_binned"] == 0, axis=1).any():
            print("Warning: Some cells have all zero counts after binning.")
    elif issparse(adata.layers["X_binned"]):
        if np.any(adata.layers["X_binned"].sum(axis=1) == 0):
             print("Warning: Some cells have all zero counts after binning (sparse).")


    all_counts_binned = adata.layers["X_binned"]
    if isinstance(all_counts_binned, torch.Tensor): # Ensure it's numpy for tokenization function
        all_counts_binned = all_counts_binned.cpu().numpy()
    if issparse(all_counts_binned):
        all_counts_binned = all_counts_binned.toarray() # Tokenizer might expect dense

    genes_for_vocab = adata.var["gene_name"].tolist()


    # VOCABULARY & MODEL INITIALIZATION
    # Try to load vocab from model_dir if load_model is specified, else create from data
    model_dir_path = None
    if args.load_model:
        model_dir_path = Path(args.load_model)
        if model_dir_path.is_dir(): # If path is a directory
            vocab_path_in_model_dir = model_dir_path / "vocab.json"
        else: # If path is a file, assume vocab is in the same dir
            vocab_path_in_model_dir = model_dir_path.parent / "vocab.json"

    if args.load_model and vocab_path_in_model_dir.exists():
        print(f"Loading vocabulary from {vocab_path_in_model_dir}")
        vocab = GeneVocab.from_file(vocab_path_in_model_dir)
        # Ensure essential tokens are present
        for s in ["<pad>", "<cls>", "<eoc>", "<mask>"]: # <mask> is often used by scGPT
            if s not in vocab:
                vocab.append_token(s)
        # Update config from loaded model's args.json if it exists
        if model_dir_path.is_dir():
            model_args_path = model_dir_path / "args.json" # scGPT saves args here
            model_config_path = model_dir_path / "config.json" # scGPT saves config here
        else:
            model_args_path = model_dir_path.parent / "args.json"
            model_config_path = model_dir_path.parent / "config.json"

        # Prioritize loaded model's config for architecture params
        loaded_model_config = {}
        if model_config_path.exists():
            print(f"Loading model architecture config from {model_config_path}")
            with open(model_config_path, "r") as f_cfg:
                loaded_model_config = json.load(f_cfg)
            # Override relevant parts of current config
            for key in ["embsize", "d_hid", "nlayers", "nhead", "dropout", "n_bins", 
                        "MVC", "ECS", "CLS", "CCE", "fast_transformer", "pre_norm"]:
                if key in loaded_model_config:
                    config[key] = loaded_model_config[key]
            if "n_batch_labels" in loaded_model_config and "num_batch_labels" not in config: # scGPT uses n_batch_labels
                 config["num_batch_labels"] = loaded_model_config["n_batch_labels"]

    else:
        print("Creating new vocabulary from training data.")
        # Use GeneVocab from scGPT
        vocab = GeneVocab(adata.var["gene_name"].tolist(), sos_token=None, eos_token=None) # No SOS/EOS for scGPT typically
        vocab.append_token("<cls>")  # Cell embedding token
        vocab.append_token("<pad>")
        vocab.append_token("<eoc>")  # End of cell token (might be useful)
        vocab.append_token("<mask>") # Mask token
        vocab.set_default_index(vocab["<pad>"])
        # Save new vocab
        vocab.to_file(Path(args.save_dir) / "vocab.json")
        print(f"Saved new vocabulary to {Path(args.save_dir) / 'vocab.json'}")


    # Tokenize data
    max_seq_len = 1200 + 1 # Max HVG + 1 for CLS, adjust based on actual gene count + CLS
    # Use actual number of genes if less than a high threshold, plus one for CLS
    max_seq_len = min(len(genes_for_vocab) + 1, 4001) # Original had 4001 for train, 3001 for test
    print(f"Using max_seq_len: {max_seq_len}")


    tokenized_data = tokenize_and_pad_batch(
        all_counts_binned,
        np.array(vocab(genes_for_vocab), dtype=int), # Gene symbols to vocab IDs
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=config["n_bins"], # scGPT uses n_bins (0 to n_bins-1 are values, n_bins is pad_value)
        append_cls=True,  # Prepend CLS token
        include_zero_gene=True, # Include all genes, even if count is zero (after binning)
        cls_appended_before = False # Append CLS at the end if False, or beginning if True (scGPT prepends)
    )

    data_pt = prepare_data(
        tokenized_data, args.species, config["mask_ratio"], 
        mask_value=vocab["<mask>"], # Use vocab's mask token ID
        pad_value=config["n_bins"], # Ensure pad_value is consistent
        batch_ids=batch_ids
    )
    
    data_loader = prepare_dataloader(
        data_pt, batch_size=args.batch_size, 
        shuffle=(args.mode == "train" and not is_distributed), # Shuffle only for train and non-DDP
        drop_last=(args.mode == "train"), # Drop last incomplete batch for training
        distributed=is_distributed
    )
    
    # MODEL
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nhead"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        nlayers_cls=3, # Number of layers for CLS head, adjust as needed
        n_cls=1, # Output classes for CLS head, not used if CLS is for embedding only
        vocab=vocab,
        dropout=config["dropout"],
        pad_token=vocab.get_stoi()["<pad>"], # Use vocab to get pad token ID
        pad_value=config["n_bins"], # Value used for padding in input
        do_mvc=config["MVC"],
        do_dab=False, # DAB is specific, keep false unless known needed
        use_batch_labels=config["INPUT_BATCH_LABELS"],
        num_batch_labels=config.get("num_batch_labels", 1),
        domain_spec_batchnorm=config["DSBN"],
        input_emb_style="continuous", # "category" or "continuous" based on scGPT
        n_input_bins=config["n_bins"], # Required for "continuous" style if values are binned indices
        cell_emb_style="cls", # Use CLS token for cell embedding
        mvc_decoder_style="innerproduct", # Or "mlp"
        ecs_threshold=0.0, # Not used if ECS is False
        explicit_zero_prob=False, # For ZINB loss, not used here
        use_fast_transformer=config["fast_transformer"],
        pre_norm=config["pre_norm"],
        use_cls_norm=True # Add LayerNorm to CLS token output
    )

    if args.load_model:
        model_load_path = Path(args.load_model)
        final_model_file_to_load = None

        if model_load_path.is_dir(): # If directory, choose model file
            # Try common scGPT model names
            if (model_load_path / "best_model.pt").exists():
                final_model_file_to_load = model_load_path / "best_model.pt"
            elif (model_load_path / "model.pt").exists():
                 final_model_file_to_load = model_load_path / "model.pt"
            # Fallback to original logic if needed
            elif args.mode == "train" and (model_load_path / "CAPTAIN_Base.pt").exists():
                final_model_file_to_load = model_load_path / "CAPTAIN_Base.pt"
            elif args.mode == "eval" and (model_load_path / "pretrain_model.pt").exists():
                final_model_file_to_load = model_load_path / "pretrain_model.pt"
            else:
                print(f"Warning: Could not find a suitable model file in {model_load_path}")
        elif model_load_path.is_file(): # If it's a file path directly
            final_model_file_to_load = model_load_path
        
        if final_model_file_to_load and final_model_file_to_load.exists():
            print(f"Loading model weights from {final_model_file_to_load}")
            try:
                saved_state_dict = torch.load(final_model_file_to_load, map_location=device)
                
                # Handle potential DDP "module." prefix
                if all(key.startswith("module.") for key in saved_state_dict.keys()):
                    print("Stripping 'module.' prefix from saved state dictionary.")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in saved_state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    saved_state_dict = new_state_dict
                
                # Handle potential "rna_model." prefix if it was part of a CombinedModel
                # This might not be needed if loading scGPT native models
                if all(key.startswith("rna_model.") for key in saved_state_dict.keys()):
                    print("Stripping 'rna_model.' prefix from saved state dictionary.")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in saved_state_dict.items():
                        if k.startswith("rna_model."): # Filter only rna_model keys
                           name = k[len("rna_model."):] 
                           new_state_dict[name] = v
                    saved_state_dict = new_state_dict

                model.load_state_dict(saved_state_dict, strict=False) # strict=False to ignore missing/extra keys
                print(f"Successfully loaded model weights from {final_model_file_to_load}")
            except Exception as e:
                print(f"Error loading model weights: {e}. Model will be randomly initialized.")
        else:
            print(f"Model file {final_model_file_to_load if final_model_file_to_load else args.load_model} not found. Model will be randomly initialized.")


    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False) # Set find_unused_parameters based on model structure

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["epochs"] // 4, gamma=0.5) # Example scheduler
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    if args.mode == "train":
        print("Starting training...")
        best_loss = float('inf')
        for epoch in range(1, config["epochs"] + 1):
            epoch_num = epoch # Set global epoch_num for logging
            if is_distributed: # Important for DDP to shuffle data correctly each epoch
                data_loader.sampler.set_epoch(epoch)

            avg_loss, avg_mlm_loss = train(model, data_loader, device, vocab, "<pad>", optimizer, scaler, config)
            print(f"Epoch {epoch:3d}/{config['epochs']:3d} | Avg Loss: {avg_loss:5.4f} | Avg MLM Loss: {avg_mlm_loss:5.4f}")
            scheduler.step()

            # Save model checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                model_save_path = Path(args.save_dir) / "best_model.pt"
                print(f"New best model with loss {best_loss:.4f}, saving to {model_save_path}")
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(), model_save_path)
            
            if epoch % 10 == 0 or epoch == config["epochs"]: # Save periodically
                 model_save_path_epoch = Path(args.save_dir) / f"epoch_{epoch}_model.pt"
                 torch.save(model.module.state_dict() if is_distributed else model.state_dict(), model_save_path_epoch)
                 print(f"Saved model checkpoint for epoch {epoch} to {model_save_path_epoch}")


    elif args.mode == "eval":
        if args.load_model is None:
            sys.exit("Evaluation mode requires --load_model to specify the model path.")
        print("Starting evaluation...")
        evaluate(model, data_loader, device, vocab, "<pad>", args.save_dir, config)
        print("Evaluation finished.")

    if is_distributed:
        dist.destroy_process_group()
    print("Script finished.")
