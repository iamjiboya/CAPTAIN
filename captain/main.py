import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import pickle as pkl
import mudata as md
import muon as mu
from config import Config, hyperparameter_defaults
from utils import seed_all, read_json_file
from data_preprocessing import our_step_preporcess
from data_preparation import prepare_data_human, prepare_data_mouse, prepare_dataloader
from models import CombinedModel
from training import train
from scgpt.tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss, quantile_loss
from scgpt.preprocess import Preprocessor
from protein_model import BLIP_Pretrain
import numpy as np
from scipy.sparse import issparse

# Global settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
MLM = True
CLS = False
ADV = False
CCE = False
INPUT_BATCH_LABELS = False
do_sample_in_train = False
per_seq_batch_sample = False

# Setup configuration and environment
config = Config(hyperparameter_defaults)
seed_all(config.seed)
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='gloo')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
seed_all(config.seed + dist.get_rank())

# Load external data
vocab_temp = read_json_file("/home/jiboya/captain/vocab.json")
with open('/home/jiboya/captain/human_mouse_align.pickle', 'rb') as fp:
    human_mouse_align = pkl.load(fp)
with open('/home/jiboya/captain/adt_token_dict.pickle', 'rb') as fp:
    adt_token_dict = pkl.load(fp)
with open('/home/jiboya/captain/adt_align_dict.pickle', 'rb') as fp:
    adt_align_dict = pkl.load(fp)

# Model initialization
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

ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    model_configs["embsize"],
    model_configs["nheads"],
    model_configs["d_hid"],
    model_configs["nlayers"],
    nlayers_cls=3,
    n_cls=1 if CLS else 1,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=-2 if config.n_bins else -1,
    do_mvc=config.MVC,
    do_dab=False,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=1,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style="continuous",
    n_input_bins=config.n_bins,
    cell_emb_style="cls",
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.include_zero_gene and MLM,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)
if config.load_model:
    try:
        model.load_state_dict({k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model')})
    except:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(model_file, map_location=device).items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
model = CombinedModel(model, adt_model)
model.to(device)
model = DDP(model, device_ids=[local_rank])

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
criterion = masked_mse_loss
criterion_quantile = quantile_loss

# Training loop
save_dir = Path("/home/jiboya/captain/")
save_dir.mkdir(parents=True, exist_ok=True)
for epoch in range(config.epochs):
    for species in ["human", "mouse"]:
        fold_fold_1 = "/home/jiboya/captain/"
        save_model_count = 0
        for file_name in os.listdir(fold_fold_1):
            save_model_count += 1
            print(f"Processing {file_name} for {species}")
            a = mu.read_h5mu(os.path.join(fold_fold_1, file_name))
            adata, adt_data = our_step_preporcess(a.mod["rna"], a.mod["adt"], species, vocab_temp, adt_token_dict, adt_align_dict)

            adata.var["gene_name"] = adata.var.index.tolist()
            adt_data.var["gene_name"] = adt_data.var.index.tolist()
            adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
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
                binning=config.n_bins,
                result_binned_key="X_binned",
            )
            preprocessor(adata, batch_key=None)

            all_counts = adata.layers["X_binned"].A if issparse(adata.layers["X_binned"]) else adata.layers["X_binned"]
            gene_ids = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)
            tokenized_train = tokenize_and_pad_batch(
                all_counts,
                gene_ids,
                max_len=3001,
                vocab=vocab,
                pad_token=pad_token,
                pad_value=-2,
                append_cls=True,
                include_zero_gene=config.include_zero_gene,
            )

            train_data_pt = (prepare_data_human if species == "human" else prepare_data_mouse)(
                tokenized_train, config.mask_ratio, -1, -2, adt_data
            )
            train_loader = prepare_dataloader(
                train_data_pt,
                config.batch_size,
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=False,
            )

            if config.do_train:
                train(model, train_loader, config, device, optimizer, scheduler, scaler, criterion, criterion_quantile, vocab, pad_token, -1)
                if save_model_count % 10 == 0:
                    torch.save(model.state_dict(), save_dir / f"{os.path.splitext(file_name)[0]}_model.pt")
                scheduler.step()