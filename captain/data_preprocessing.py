import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys

def preprocss_rna(data, species, vocab_temp):
    """Preprocess RNA data."""
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

def preprocss_adt(data, species, adt_token_dict, adt_align_dict):
    """Preprocess ADT data."""
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
    """Check AnnData X matrix for negative or float values."""
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

def our_step_preporcess(adata, adata_protein, species, vocab_temp, adt_token_dict, adt_align_dict):
    """Perform preprocessing steps for RNA and ADT data."""
    check_adata_x(adata)
    check_adata_x(adata_protein)
    rna_data_pre = preprocss_rna(adata, species=species, vocab_temp=vocab_temp)
    adt_data_pre = preprocss_adt(adata_protein, species=species, adt_token_dict=adt_token_dict, adt_align_dict=adt_align_dict)
    common_obs = rna_data_pre.obs_names.intersection(adt_data_pre.obs_names)
    rna_data_pre = rna_data_pre[common_obs]
    adt_data_pre = adt_data_pre[common_obs]
    return rna_data_pre, adt_data_pre