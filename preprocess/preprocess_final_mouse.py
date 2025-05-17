import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import pickle as pkl
import mudata as md
from mudata import MuData
import muon as mu
import os,itertools
import math,json,scipy,sys
import scipy.sparse as sp

# Function to read JSON files
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
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index
    rna_name = data.var.index.tolist()
    common_elements = set(rna_name) & set(vocab_temp.keys())
    print("Number of RNA genes present in both the RNA list and AnnData object:", len(common_elements))
    if len(common_elements) == 0:
        print("No matching genes found, exiting the program.")
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
    print("Number of ADT genes present in both the ADT list and AnnData object:", len(common_elements))
    if len(common_elements) == 0:
        print("No matching proteins found, exiting the program.")
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

fold_fold_1 = "/home/jiboya/captain/mouse/"
fold_fold_2 = "/home/jiboya/captain/mouse_done/"
species = "mouse"
for file_name in os.listdir(fold_fold_1):

    print(file_name)
    a = mu.read_h5mu(fold_fold_1 + file_name)
    adata = a.mod["rna"]
    adata_protein = a.mod["adt"]
    adata, adata_protein = our_step_preporcess(adata, adata_protein, species)
    print(adata)
    print(adata_protein)
    mdata = mu.MuData({"rna": adata, "adt": adata_protein})
    mdata.write_h5mu(fold_fold_2 + file_name, compression="gzip")
