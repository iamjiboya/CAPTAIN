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
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, quantile_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

vocab_temp = read_json_file("/home/jiboya/captain/vocab.json")

with open('/home/jiboya/captain/human_mouse_align.pickle', 'rb') as fp:
    human_mouse_align = pkl.load(fp)
with open('/home/jiboya/captain/adt_token_dict.pickle', 'rb') as fp:
    adt_token_dict = pkl.load(fp)
with open('/home/jiboya/captain/adt_align_dict.pickle', 'rb') as fp:
    adt_align_dict = pkl.load(fp)

def preprocss_rna(data, species):
    sc.pp.filter_genes(data, min_counts=10)
    sc.pp.filter_cells(data, min_counts=200)
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index
    rna_name = data.var.index.tolist()
    common_elements = set(rna_name) & set(vocab_temp.keys())
    print("RNA gene list presence in AnnData object", len(common_elements))
    if len(common_elements) == 0:
        print("No matching genes found, exiting program.")
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
    print("ADT gene list presence in AnnData object", len(common_elements))
    if len(common_elements) == 0:
        print("No matching proteins found, exiting program.")
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
    if scipy.sparse.issparse(adata.X):
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

def prepare_data_mouse(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
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

def prepare_data_human(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
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

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
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
        return data_loader
    dataset_sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        sampler=dataset_sampler
    )
    return data_loader

def train(model: nn.Module, loader: DataLoader) -> None:
    model.train()
    dist.barrier()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict, transformer_out = model.module.rna_model(
                input_gene_ids,
                input_values,
                species_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )
            adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.module.adt_model(
                adt_data,
                transformer_out,
                src_key_padding_mask,
                adt_values,
                output_atten=False
            )
            masked_positions = input_values.eq(mask_value)
            loss = 0.0
            metrics_to_log = {}
            loss_weights = {
                'mlm': 0.2,
                "adt_mse": 0.6,
                "adt_quantile": 0.2,
            }
            loss_mlm = loss_weights["mlm"] * criterion(output_dict["mlm_output"], target_values, masked_positions)
            loss_adt_mse = loss_weights["adt_mse"] * criterion(adt_to_out.squeeze(-1), labels_adt_data, adt_mask)
            loss_adt_quantile = loss_weights["adt_quantile"] * criterion_quantile(adt_to_out_quantiles, labels_adt_data, adt_mask)
            loss = loss_mlm + loss_adt_mse + loss_adt_quantile
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_mse += loss_mlm.item()
        total_cls += loss_adt_mse.item()
        total_cce += loss_adt_quantile.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval
            cur_cce = total_cce / log_interval
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            cur_mvc_zero_log_prob = total_mvc_zero_log_prob / log_interval if MVC and explicit_zero_prob else 0.0
            cur_error = 0
            print(
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                f"cls {cur_cls:5.2f} | "
                f"err {cur_error:5.2f} | " if CLS else ""
                f"cce {cur_cce:5.2f} |"
                f"mvc {cur_mvc:5.2f} |" if MVC else ""
                f"ecs {cur_ecs:5.2f} |" if ECS else ""
                f"dab {cur_dab:5.2f} |" if DAB else ""
                f"adv_E {cur_adv_E:5.2f} |" if ADV else ""
                f"adv_D {cur_adv_D:5.2f} |" if ADV else ""
                f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else ""
                f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |" if MVC and explicit_zero_prob else ""
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()

class Config:
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key, value)

def seed_all(seed_value, cuda_deterministic=False):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class CombinedModel(nn.Module):
    def __init__(self, main_model, sub_model):
        super(CombinedModel, self).__init__()
        self.rna_model = main_model
        self.adt_model = sub_model

    def forward(self, x):
        pass

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="/home/jiboya/captain",
    mask_ratio=0.15,
    epochs=40,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=1.0,
    lr=1e-5,
    batch_size=20,
    layer_size=512,
    nlayers=12,
    nhead=8,
    dropout=0.2,
    schedule_ratio=1,
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
print(config)

set_seed(config.seed)
local_rank = int(os.environ['LOCAL_RANK'])
is_master = local_rank == 0
dist.init_process_group(backend='gloo')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
seed_all(config.seed + torch.distributed.get_rank())

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = 3001
n_bins = config.n_bins

input_style = "binned"
output_style = "binned"
MLM = True
CLS = False
ADV = False
CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "cls"
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

lr = config.lr
lr_ADV = 1e-3
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout

log_interval = 100
save_eval_interval = config.save_eval_interval
do_eval_scib_metrics = True

assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError("input_emb_style `category` is not supported for log1p or normed_raw input.")

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

dataset_name = config.dataset_name
save_dir = Path(f"/home/jiboya/captain/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(f"Resume model from {model_file}, the model args will override the config {model_config_file}.")
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=1 if CLS else 1,
    vocab=vocab,
    dropout=dropout,
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
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        rna_model_state_dict = {
            k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model')
        }
        model.load_state_dict(rna_model_state_dict)
        print(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
for name, para in model.named_parameters():
    print("-"*20)
    print(f"name: {name}")
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
print(f"Total Pre freeze Params {(pre_freeze_param_count )}")
print(f"Total Post freeze Params {(post_freeze_param_count )}")
print({"info/pre_freeze_param_count": pre_freeze_param_count, "info/post_freeze_param_count": post_freeze_param_count})

from protein_model import BLIP_Pretrain
print("Creating model")
adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
#adt_model_state_dict = {
#    k[len('module.adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.adt_model')
#}
#adt_model.load_state_dict(adt_model_state_dict)

model = CombinedModel(model, adt_model)
model.to(device)
model = DDP(model, device_ids=[local_rank])

if ADV:
    discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=1).to(device)

criterion = masked_mse_loss
criterion_quantile = quantile_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=config.schedule_ratio)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(optimizer_dab, schedule_interval, gamma=config.schedule_ratio)
if ADV:
    criterion_adv = nn.CrossEntropyLoss()
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, schedule_interval, gamma=config.schedule_ratio)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, schedule_interval, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None

for k in range(epochs):
    fold_fold_1 = "/home/jiboya/captain/"
    species = "human"
    save_model_count = 0
    for file_name in os.listdir(fold_fold_1):
        save_model_count += 1
        print(file_name)
        a = mu.read_h5mu(fold_fold_1 + file_name)
        adata = a.mod["rna"]
        adata_protein = a.mod["adt"]
        adata.var.set_index(adata.var.index, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
        adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
        old_vocab = vocab
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
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,
            result_binned_key="X_binned",
        )
        preprocessor(adata, batch_key=None)
        input_layer_key = {"normed_raw": "X_normed", "log1p": "X_normed", "binned": "X_binned"}[input_style]
        all_counts = adata.layers[input_layer_key].A if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
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
        if species == "human":
            train_data_pt = prepare_data_human(sort_seq_batch=per_seq_batch_sample)
        if species == "mouse":
            train_data_pt = prepare_data_mouse(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        if config.do_train:
            train(model, loader=train_loader)
            if save_model_count % 10 == 0:
                name_without_ext = os.path.splitext(file_name)[0]
                name = str(save_dir) + "/" + str(name_without_ext) + "_model.pt"
                torch.save(model.state_dict(), name)
            scheduler.step()
            if DAB_separate_optim:
                scheduler_dab.step()
            if ADV:
                scheduler_D.step()
                scheduler_E.step()

    fold_fold_1 = "/home/jiboya/captain/"
    species = "mouse"
    save_model_count = 0
    for file_name in os.listdir(fold_fold_1):
        save_model_count += 1
        print(file_name)
        a = mu.read_h5mu(fold_fold_1 + file_name)
        adata = a.mod["rna"]
        adata_protein = a.mod["adt"]
        adata.var.set_index(adata.var.index, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
        adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
        old_vocab = vocab
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
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,
            result_binned_key="X_binned",
        )
        preprocessor(adata, batch_key=None)
        input_layer_key = {"normed_raw": "X_normed", "log1p": "X_normed", "binned": "X_binned"}[input_style]
        all_counts = adata.layers[input_layer_key].A if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
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
        if species == "human":
            train_data_pt = prepare_data_human(sort_seq_batch=per_seq_batch_sample)
        if species == "mouse":
            train_data_pt = prepare_data_mouse(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        if config.do_train:
            train(model, loader=train_loader)
            if save_model_count % 5 == 0:
                name_without_ext = os.path.splitext(file_name)[0]
                name = str(save_dir) + "/" + str(name_without_ext) + "_model.pt"
                torch.save(model.state_dict(), name)
            scheduler.step()
            if DAB_separate_optim:
                scheduler_dab.step()
            if ADV:
                scheduler_D.step()
                scheduler_E.step()