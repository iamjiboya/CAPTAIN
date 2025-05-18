# CAPTAIN: A multimodal foundation model pretrained on co-assayed single-cell RNA and protein
===========================================================================


[![license](https://img.shields.io/badge/python_-3.9.1_-brightgreen)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.1.2_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.10.4_-purple)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.11.3_-red)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/muon_-0.1.7_-yellow)](https://muon-tutorials.readthedocs.io/en/latest/index.html)
[![license](https://img.shields.io/badge/omicverse_-1.6.10_-lime)](https://omicverse.readthedocs.io/)
[![license](https://img.shields.io/badge/R_-4.2.2_-success)](https://www.r-project.org/)

Proteins act as the ultimate executors of cellular function, encoding the phenotypic consequences of genomic and transcriptomic programs. Although transcriptomic profiles serve as accessible proxies, they remain incomplete surrogates for the proteomic landscape that ultimately defines cellular phenotypes. Current single-cell foundation models, however, are trained exclusively on transcriptomes, resulting in biased and partial characterizations of cellular states. To address this limitation, we introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 387 surface proteins across diverse human and mouse tissues. Our results show that CAPTAIN learns unified multimodal representations by modeling cross-modality dependencies and capturing the diversity of cellular states across complex biological contexts. CAPTAIN generalizes robustly across both fine-tuning and zero-shot settings, excelling in core downstream tasks such as protein imputation, cell type annotation, and batch harmonization. Beyond improved accuracy in multi-omics integration, CAPTAIN uncovers previously inaccessible mechanisms of protein-mediated intercellular dynamics, including immune interaction patterns linked to COVID-19 severity. CAPTAIN establishes a new paradigm for multimodal single-cell modeling, laying the foundation for comprehensive cellular understanding and virtual cell construction.
![Image text](https://github.com/iamjiboya/CAPTAIN/blob/main/img/CAPTAIN.png)

## Installation

CAPTAIN is implemented based on Pytorch. We use pytorch-2.1.2 and cuda-12.8. Other version could be also compatible. We highly recommend using Anaconda to manage your Python environment. This ensures a consistent and reproducible setup for running our model. To create the recommended environment, please follow these steps:

1.  **Install Anaconda:** If you haven't already, download and install Anaconda from the official website: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

2.  **Create the environment:** Navigate to the root directory of this repository in your terminal or Anaconda Prompt. Then, execute the following command to create the environment based on the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

    This command will create a new conda environment with all the necessary dependencies and configurations as specified in the `environment.yml` file, guaranteeing a complete and reproducible environment for optimal performance of the model.

3.  **Activate the environment:** Once the environment is created, activate it using the following command:

    ```bash
    conda activate captain
    ```
Alternatively, for users who prefer using pip, we have also included a `requirements.txt` file. This file lists the necessary Python packages required to run the model.

    
    pip install -r requirements.txt
    
**Notes on FlashAttention for Accelerated Training:**


For accelerated training, we also leverage FlashAttention. We specifically utilize FlashAttention-2 and recommend CUDA 12.8 for optimal performance.

Please be aware that the `flash-attn` dependency often requires specific GPU hardware and CUDA versions. Therefore, for detailed and up-to-date installation instructions tailored to your system, please refer directly to the official `flash-attn` repository: https://github.com/Dao-AILab/flash-attention/tree/main.

## Pretrained CAPTAIN Models

We introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 387 surface proteins across diverse human and mouse tissues. You can download the pretrained model checkpoints below. Place the downloaded model directory in the main path (e.g., `./pretrained_models/CAPTAIN_Base`, `./pretrained_models/CAPTAIN_PBMC`, `./pretrained_models/CAPTAIN_Human`).

| Model             | Description                                                                                                                                                                                             | Download |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `CAPTAIN_Base`    | Base model pretrained on the complete dataset, encompassing diverse human and mouse tissues.                                                                                                          | [Link](YOUR_CAPTAIN_BASE_DOWNLOAD_LINK_HERE)    |
| `CAPTAIN_PBMC`    | Model specifically pretrained on Peripheral Blood Mononuclear Cell (PBMC) data, the most commonly profiled tissue in single-cell multiomics sequencing. | [Link](YOUR_CAPTAIN_PBMC_DOWNLOAD_LINK_HERE)    |




## Token Dictionaries

The following table describes the token dictionary files included in this repository, which are essential for understanding the input and output representations of our model(You can find these files in the [token_dict](https://github.com/iamjiboya/CAPTAIN/blob/main/token_dict) folder):

| Filename               | Description                                                                                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `human_mouse_align.pickle` | Provides a mapping for converting gene names between human and mouse, crucial for cross-species analyses and comparisons.                                          |
| `csp_align_dict.pickle`  | Facilitates the alignment and conversion of cell surface protein names to our standardized nomenclature used within the model.                                |
| `csp_token_dict.pickle`  | Contains the vocabulary for the 387 cell surface proteins used in our model, mapping each protein name to its corresponding numerical ID within the token space. |
| `vocab.json`           | Serves as the vocabulary for gene names, mapping gene symbols to their respective numerical IDs within the token space. Derived from scGPT, encompassing 60,693 human genes. |

## Data Preprocessing


The data preprocessing steps applied to the multimodal single-cell RNA and Protein data. The preprocessing pipeline reads a MuData object, processes the RNA and Protein modalities independently, ensures that only cells with measurements in both modalities are retained, and then saves the processed MuData object (You can find these files in the [preprocess](https://github.com/iamjiboya/CAPTAIN/blob/main/preprocess) folder).

## Pre-training

Our model undergoes a pre-training phase to learn foundational representations from large-scale multimodal single-cell datasets (You can find these files in the [pretrain](https://github.com/iamjiboya/CAPTAIN/blob/main/pretrain) folder). To accelerate this computationally intensive process, we employ **parallel training** across multiple GPUs.

The transcriptional module within our model is initialized with weights from the pre-trained scGPT model. To begin, please download the scGPT model files as they are necessary for initializing the transcriptional component.

**Running Pre-training:**

The pre-training process can be initiated using the following command as an example. This command utilizes `torchrun` for distributed training:

```bash
torchrun --nproc_per_node=4 --master_port=29512 /home/jiboya/captain/pretrain/torchrun.py --gpu 0,1,2,3
