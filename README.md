# CAPTAIN: A multimodal foundation model pretrained on co-assayed single-cell RNA and protein
===========================================================================


[![license](https://img.shields.io/badge/python_-3.9.1_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.1.2_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.10.4_-purple)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.11.3_-red)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/muon_-0.1.7_-yellow)](https://muon-tutorials.readthedocs.io/en/latest/index.html)
[![license](https://img.shields.io/badge/omicverse_-1.6.10_-lime)](https://omicverse.readthedocs.io/)
[![license](https://img.shields.io/badge/R_-4.2.2_-success)](https://www.r-project.org/)


!UPDATE: We have released all datasets used for pre-training. These datasets can be further explored, searched, and downloaded from our dedicated data portal at: [scT&P-4M](https://sctp4m.aigenomicsyulab.com).


Proteins act as the terminal effectors of cellular function, encoding the phenotypic consequences of genomic and transcriptomic programs. Although transcriptomic profiles serve as accessible proxies, they remain incomplete surrogates for the proteomic landscape that ultimately defines cellular phenotypes. Current single-cell foundation models, however, are trained exclusively on transcriptomes, resulting in biased and partial characterizations of cellular states. To address this limitation, we introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 382 surface proteins across diverse human and mouse tissues. Our results show that CAPTAIN learns unified multimodal representations by modeling cross-modality dependencies and capturing the diversity of cellular states across complex biological contexts. CAPTAIN generalizes robustly across both fine-tuning and zero-shot settings, excelling in core downstream tasks such as protein imputation and expansion, cell type annotation, and batch harmonization. Beyond improved accuracy in multi-omics integration, CAPTAIN uncovers previously inaccessible mechanisms of protein-driven intercellular dynamics, including immune interaction patterns linked to COVID-19 severity. CAPTAIN establishes a new paradigm for multimodal single-cell modeling, laying the foundation for comprehensive cellular understanding and virtual cell construction.
![Image text](https://github.com/iamjiboya/CAPTAIN/blob/main/img/CAPTAIN.png)

## Installation

CAPTAIN is implemented based on Pytorch. We use pytorch-2.1.2 and cuda-12.8. Other version could be also compatible. We highly recommend using Anaconda to manage your Python environment. This ensures a consistent and reproducible setup for running our model. The environment for CAPTAIN can be obtained from the Aliyun Docker Hub registry or by installing the dependencies with requirement.txt.

**Option 1: Download the docker image from Docker Hub. （highly recommend）**

```bash
    docker pull crpi-nzg91d1psypntvav.cn-beijing.personal.cr.aliyuncs.com/jiboya/captain_image:latest
 ```
Start a container based on the image and activate the enviroment:

```bash
    docker run --gpus all -it --rm crpi-nzg91d1psypntvav.cn-beijing.personal.cr.aliyuncs.com/jiboya/captain_image:latest /bin/bash
 ```
```bash
    conda activate captain
 ```

**Option 2: Utilize conda to create and activate a environment. To create the recommended environment, please follow these steps:**

1.  **Install Anaconda:** If you haven't already, download and install Anaconda from the official website: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

2.  **Create the environment:** Navigate to the root directory of this repository in your terminal or Anaconda Prompt. Then, execute the following command to create the CAPTAIN environment:

    ```bash
    conda create -n captain python==3.10.0
    ```

3.  **Activate the environment:** Once the environment is created, activate it using the following command:

    ```bash
    conda activate captain
    ```
4.  **Install requried packages:** We have included a `requirements.txt` file. This file lists the necessary Python packages required to run the model. Among these, [scgpt](https://github.com/bowang-lab/scGPT) is a required package as it is used for initializing the model with pre-trained weights.

    ```bash
    pip install -r requirements.txt && pip install scgpt
    ```

    
**Notes on FlashAttention for Accelerated Training:**


For accelerated training, we also leverage FlashAttention. We specifically utilize FlashAttention-2 and recommend CUDA 12.8 for optimal performance.

Please be aware that the `flash-attn` dependency often requires specific GPU hardware and CUDA versions. Therefore, for detailed and up-to-date installation instructions tailored to your system, please refer directly to the official `flash-attn` repository: https://github.com/Dao-AILab/flash-attention/tree/main.

Please note that FlashAttention is optional; the model can run normally without it.

## Pretrained CAPTAIN Models

We introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 387 surface proteins across diverse human and mouse tissues. You can download the pretrained model checkpoints below. Place the downloaded model directory in the main path (e.g., `./pretrained_models/CAPTAIN_Base`, `./pretrained_models/CAPTAIN_PBMC`, `./pretrained_models/CAPTAIN_Human`).

| Model             | Description                                                                                                                                                                                             | Download |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `CAPTAIN_Base`    | Base model pretrained on the complete dataset, encompassing diverse human and mouse tissues.                                                                                                          | [Link](https://drive.google.com/drive/folders/1NE_MJQw6BliExH0l5OcpoJWe3MSJSgin?usp=drive_link)    |
| `CAPTAIN_PBMC`    | Model specifically pretrained on Peripheral Blood Mononuclear Cell (PBMC) data, the most commonly profiled tissue in single-cell multiomics sequencing. | [Link](https://drive.google.com/drive/folders/1qwQlmT2kg2-o9qwuVlNRVUzAVuwZtUXx?usp=drive_link)    |




## Token Dictionaries

The following table describes the token dictionary files included in this repository, which are essential for understanding the input and output representations of our model(You can find these files in the [token_dict](https://github.com/iamjiboya/CAPTAIN/blob/main/token_dict) folder):

| Filename               | Description                                                                                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `human_mouse_align.pickle` | Provides a mapping for converting gene names between human and mouse, crucial for cross-species analyses and comparisons.                                          |
| `csp_align_dict.pickle`  | Facilitates the alignment and conversion of cell surface protein names to our standardized nomenclature used within the model.                                |
| `csp_token_dict.pickle`  | Contains the vocabulary for the 387 cell surface proteins used in our model, mapping each protein name to its corresponding numerical ID within the token space. |
| `vocab.json`           | Serves as the vocabulary for gene names, mapping gene symbols to their respective numerical IDs within the token space. Derived from scGPT, encompassing 60,693 human genes. |


## Prior Knowledge

Building upon the prior knowledge resources offered in the original [GeneCompass](https://github.com/xCompass-AI/GeneCompass), we have processed and refined these resources to provide species-specific gene prior knowledge for both human and mouse (You must first download the following files and place them in the [prior_knowledge](https://github.com/iamjiboya/CAPTAIN/blob/main/prior_knowledge) folder).

The following files contain the processed prior knowledge:

| Filename                      | Description                                                                                                                              | Download |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `final_human_prior_knwo.npy`        | Contains processed gene prior knowledge specific to the human genome. | [Link](https://drive.google.com/file/d/1gnrq4UDhlBM9Hf8heo--RH7exwP7IUB0/view?usp=drive_link)    |
| `final_mouse_prior_knwo.npy`        | Contains processed gene prior knowledge specific to the mouse genome. | [Link](https://drive.google.com/file/d/18Sm69KL_LX8fOdDrMtFZE2hkuDZDhPS-/view?usp=drive_link)    |



## Data Preprocessing


The data preprocessing steps applied to the multimodal single-cell RNA and Protein data. The preprocessing pipeline reads a MuData object, processes the RNA and Protein modalities independently, ensures that only cells with measurements in both modalities are retained, and then saves the processed MuData object (You can find these files in the [preprocess](https://github.com/iamjiboya/CAPTAIN/blob/main/preprocess) folder).

## Pre-training

Our model undergoes a pre-training phase to learn foundational representations from large-scale multimodal single-cell datasets (You can find these files in the [pretrain](https://github.com/iamjiboya/CAPTAIN/blob/main/pretrain) folder). To accelerate this computationally intensive process, we employ **parallel training** across multiple GPUs.

The transcriptional module within our model is initialized with weights from the pre-trained scGPT model. To begin, please download the scGPT model files as they are necessary for initializing the transcriptional component.

**Running Pre-training:**

The pre-training process can be initiated using the following command as an example. This command utilizes `torchrun` for distributed training:

```bash
cd /home/captain/pretrain && torchrun --nproc_per_node=4 --master_port=29512 torchrun.py --gpu 0,1,2,3
```
## Downstream Tasks
### Fine-tuning on Pre-trained Model for Cell Surface Protein Imputation
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_surface_protein_prediction), we demonstrate how to fine-tune the pre-trained model on a new dataset for the cell surface protein prediction and imputation task. We randomly split a citeseq data of [Seurat 4 processed human peripheral blood mononuclear cells dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) into a training dataset for fine-tuning and a test dataset for prediction. Please download the dataset, fine-tuned models, generated feature files and predicted results from [Link](https://drive.google.com/drive/folders/1vT4mYzU5IPYFrD_rWtksK2DVdn9nsnQH?usp=drive_link).
### Zero-shot with Pre-trained Model for Cell Surface Protein Prediction
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_surface_protein_prediction), we demonstrate how to zero shot with the pre-trained model on a new dataset for the cell surface protein prediction and imputation task. We use the test half of [Seurat 4 processed human peripheral blood mononuclear cells dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) as an example. Please download the dataset, generated feature files and predicted results from [Link](https://drive.google.com/drive/folders/1vT4mYzU5IPYFrD_rWtksK2DVdn9nsnQH?usp=drive_link).
### Fine-tuning on Pre-trained Model for Cell-type Annotation
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_type_annotation), we demonstrate how to fine-tune the pre-trained model on a new dataset for the cell type annotation task. We use the [Seurat 4 processed human peripheral blood mononuclear cells dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) as an example. Please download the dataset, fine-tuned models, generated feature files and predicted results from [Link](https://drive.google.com/drive/folders/1Gn7S4ERAJybhn28XMIDRhAtsRVo8cG65?usp=drive_link).
### Fine-tuning on Pre-trained Model for Multiomics Integrate
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/multiomics_integrate), we demonstrate how to fine-tune the pre-trained model on a new dataset for the multiomics integrate task. We use a [Sanger COVID dataset](https://covid19.cog.sanger.ac.uk/submissions/release2/vento_pbmc_processed.h5ad) as an example. Please download the dataset, fine-tuned models, generated feature files from [Link](https://drive.google.com/drive/folders/1EUjRZqNOFYNwBpGZoAGeyccqk3couReK?usp=drive_link).
### Fine-tuning on Pre-trained Model for Batch Correct
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/batch_correct), we demonstrate how to fine-tune the pre-trained model on a new dataset for the batch correct task. We use the scRNA-seq data of [Seurat 4 processed human peripheral blood mononuclear cells dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) as an example. Please download the dataset, fine-tuned models, generated feature files and corrected results from [Link](https://drive.google.com/drive/folders/1D05clgX57ISVdUozEz-HiQu5t5kEuNpW?usp=drive_link).
### Fine-tuning on Pre-trained Model for Perturb Protein Prediction
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/perturb_protein_prediction), we demonstrate how to fine-tune the pre-trained model on a new dataset for the perturb protein prediction. We randomly split a [perturb-cite-seq data](https://www.nature.com/articles/s41588-021-00779-1) into a training dataset for fine-tuning and a test dataset for prediction. Please download the dataset, fine-tuned models, generated feature files and predicted results from [Link](https://drive.google.com/drive/folders/1c8nlX3lOUVTn75ijhTeAaD9ZW1VBGvn4).
### Fine-tuning on Pre-trained Model for Cell-Cell Communication
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_cell_communication), we demonstrate how to fine-tune the pre-trained model on a new dataset for the cell-cell communication task. We use a [CITE-Seq dataset from 10X](https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.2/5k_pbmc_protein_v3) as an example. Please download the dataset, fine-tuned models, generated feature files from [Link](https://drive.google.com/drive/folders/1wV8zkBePlZiPpAlAx_xssUkmWYpCzhuT?usp=drive_link).
### Zero-shot with Pre-trained Model for Cell-Cell Communication
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_cell_communication), we demonstrate how to zero shot with pre-trained model on a new dataset for the cell-cell communication task. We use the [scvi.data.pbmc_dataset](https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_dataset.html) (10k scRNA-seq data from two batches of peripheral blood mononuclear cells from a healthy donor) as an example. Please download the dataset and generated feature files from [Link](https://drive.google.com/drive/folders/1wV8zkBePlZiPpAlAx_xssUkmWYpCzhuT?usp=drive_link).
#### Important Note
Cell-cell communication inference relies on the predicted expression of cell surface proteins. Before proceeding with communication analysis, please first refer to the example code provided in the "Cell Surface Protein Prediction and Imputation" section. This will allow you to generate the necessary cell surface protein expression data corresponding to your scRNA-seq dataset.

## Copyright Notice
### Code License

This repository's source code is licensed under the MIT License.
### Third-party Software License

Use of the third-party software, libraries or code referred to in the Acknowledgements section may be governed by separate terms and conditions or license provisions.

Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## Reference

This project utilizes and/or references the following libraries and packages:

* scGPT
* scFoundation
* GeneCompass
* scBert
* liana+
* omicverse

## Acknowledgements

We would like to thank the contributors and maintainers of the following libraries and packages that CAPTAIN uses and/or references:


* muon
* mudata
* scanpy
* anndata
* flash-attention
* scvi-tools
* torch
* torchrun
* r
* transformers

