import json
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA

def process_gene_features(gene_list: dict, feature_dict: dict, feature_dim: int = 512) -> np.ndarray:
    """
    Returns a 2D NumPy array based on the gene list, retaining features from the feature dictionary.
    Sets features to zero vectors for genes not found in the dictionary.

    :param gene_list: Dictionary containing gene names as keys and their sorted index as values.
    :param feature_dict: Dictionary containing gene features, with gene names as keys and feature vectors as values.
    :param feature_dim: Dimensionality of the feature vectors.
    :return: A 2D NumPy array of shape (n_genes, feature_dim).
    """
    n_genes = len(gene_list)
    feature_matrix = np.zeros((n_genes, feature_dim))
    for gene, index in gene_list.items():
        if gene in feature_dict:
            feature_matrix[index] = feature_dict[gene]
        else:
            feature_matrix[index] = np.zeros(feature_dim)
    return feature_matrix

def pca_dec(data_dict: dict, n_components: int = 512) -> dict:
    """
    Applies PCA to reduce the dimensionality of feature vectors in a dictionary.

    :param data_dict: Dictionary with gene names as keys and feature vectors as values.
    :param n_components: Number of principal components to keep.
    :return: Dictionary with gene names as keys and reduced feature vectors as values.
    """
    genes = list(data_dict.keys())
    features = np.array(list(data_dict.values()))
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    reduced_data_dict = {gene: reduced_features[i] for i, gene in enumerate(genes)}
    return reduced_data_dict

def read_json_file(file_path: str) -> dict:
    """
    Reads a JSON file and returns the data as a dictionary.

    :param file_path: Path to the JSON file.
    :return: Dictionary containing the JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def dict_align(dict_in: dict, alignment_dict: dict) -> dict:
    """
    Aligns the keys of a dictionary based on a provided alignment dictionary.

    :param dict_in: Input dictionary to align.
    :param alignment_dict: Dictionary for alignment, mapping keys from dict_in to new keys.
    :return: A new dictionary with aligned keys.
    """
    dict_out = {}
    for key, value in dict_in.items():
        try:
            dict_out[alignment_dict[key]] = value
        except KeyError:
            continue
    return dict_out

# Load human-mouse gene alignment
with open('/home/jiboya/captain/human_mouse_align.pickle', 'rb') as fp:
    human_mouse_align = pkl.load(fp)

# Load human gene names
human_gene_name = read_json_file("/home/jiboya/captain/vocab.json")

# Load human prior knowledge embeddings
with open('/home/jiboya/captain/Human_dim_768_gene_28291_random.pickle', 'rb') as fp:
    human_coexpress_emb = pkl.load(fp)
with open('/home/jiboya/captain/Human_dim_768_gene_28291_random_2.pickle', 'rb') as fp:
    human_gene_family_emb = pkl.load(fp)
with open('/home/jiboya/captain/human_PECA_vec.pickle', 'rb') as fp:
    human_peca_emb = pkl.load(fp)
with open('/home/jiboya/captain/human_emb_768.pickle', 'rb') as fp:
    human_promoter_emb = pkl.load(fp)

# Reduce dimensionality of human embeddings
human_coexpress_emb_pca = pca_dec(human_coexpress_emb)
human_gene_family_emb_pca = pca_dec(human_gene_family_emb)
human_peca_emb_pca = pca_dec(human_peca_emb)
human_promoter_emb_pca = pca_dec(human_promoter_emb)

# Process human prior knowledge
human_coexpress_processed = process_gene_features(human_gene_name, human_coexpress_emb_pca)
human_gene_family_processed = process_gene_features(human_gene_name, human_gene_family_emb_pca)
human_peca_processed = process_gene_features(human_gene_name, human_peca_emb_pca)
human_promoter_processed = process_gene_features(human_gene_name, human_promoter_emb_pca)

# Combine human prior knowledge
human_prior_knowledge = np.concatenate([human_coexpress_processed, human_gene_family_processed, human_peca_processed, human_promoter_processed], axis=1)
np.save("/home/jiboya/captain/human_prior_knwo.npy", human_prior_knowledge)

# Load mouse prior knowledge embeddings
with open('/home/jiboya/captain/Mouse_dim_768_gene_27444_random.pickle', 'rb') as fp:
    mouse_coexpress_emb = pkl.load(fp)
with open('/home/jiboya/captain/Mouse_dim_768_gene_27934_random.pickle', 'rb') as fp:
    mouse_gene_family_emb = pkl.load(fp)
with open('/home/jiboya/captain/mouse_PECA_vec.pickle', 'rb') as fp:
    mouse_peca_emb = pkl.load(fp)
with open('/home/jiboya/captain/mouse_emb_768.pickle', 'rb') as fp:
    mouse_promoter_emb = pkl.load(fp)

# Align and reduce dimensionality of mouse embeddings
mouse_coexpress_emb_aligned = dict_align(mouse_coexpress_emb, human_mouse_align)
mouse_gene_family_emb_aligned = dict_align(mouse_gene_family_emb, human_mouse_align)
mouse_peca_emb_aligned = dict_align(mouse_peca_emb, human_mouse_align)
mouse_promoter_emb_aligned = dict_align(mouse_promoter_emb, human_mouse_align)

mouse_coexpress_emb_pca = pca_dec(mouse_coexpress_emb_aligned)
mouse_gene_family_emb_pca = pca_dec(mouse_gene_family_emb_aligned)
mouse_peca_emb_pca = pca_dec(mouse_peca_emb_aligned)
mouse_promoter_emb_pca = pca_dec(mouse_promoter_emb_aligned)

# Process mouse prior knowledge
mouse_coexpress_processed = process_gene_features(human_gene_name, mouse_coexpress_emb_pca)
mouse_gene_family_processed = process_gene_features(human_gene_name, mouse_gene_family_emb_pca)
mouse_peca_processed = process_gene_features(human_gene_name, mouse_peca_emb_pca)
mouse_promoter_processed = process_gene_features(human_gene_name, mouse_promoter_emb_pca)

# Combine mouse prior knowledge
mouse_prior_knowledge = np.concatenate([mouse_coexpress_processed, mouse_gene_family_processed, mouse_peca_processed, mouse_promoter_processed], axis=1)
np.save("/home/jiboya/captain/mouse_prior_knwo.npy", mouse_prior_knowledge)