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

Building upon the prior knowledge resources offered in the original [GeneCompass](https://github.com/xCompass-AI/GeneCompass), we have processed and refined these resources to provide species-specific gene prior knowledge for both human and mouse (You can find these files in the [prior_knowledge](https://github.com/iamjiboya/CAPTAIN/blob/main/prior_knowledge) folder).

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
torchrun --nproc_per_node=4 --master_port=29512 /home/jiboya/captain/pretrain/torchrun.py --gpu 0,1,2,3
```
## Downstream Tasks
### Fine-tuning on Pre-trained Model for Cell-type Annotation
In this [tutorial](https://github.com/iamjiboya/CAPTAIN/tree/main/downstream_tasks/cell_type_annotation), we demonstrate how to fine-tune the pre-trained model on a new dataset for the cell type annotation task. We use the Seurat 4 processed human peripheral blood mononuclear cells (PBMCs) dataset (GEO: GSE164378) as an example. Please download the dataset, fine-tuned models, generated feature files and predicted results from [Link](https://drive.google.com/drive/folders/1Gn7S4ERAJybhn28XMIDRhAtsRVo8cG65?usp=drive_link).

## Copyright Notice
### Code License

This repository's source code is licensed under the Apache-2.0 License. However, please note that any use or exploitation of the Model Weights is subject to a separate Model License.
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

