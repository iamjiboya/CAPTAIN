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

# Installation

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

## Download Checkpoints

We introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 387 surface proteins across diverse human and mouse tissues. You can download the pretrained model checkpoints below. Place the downloaded model directory in the main path (e.g., `./pretrained_models/CAPTAIN_Base`, `./pretrained_models/CAPTAIN_PBMC`, `./pretrained_models/CAPTAIN_Human`).

| Model             | Description                                                                                                                                                                                             | Download |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `CAPTAIN_Base`    | Base model pretrained on the complete dataset, encompassing diverse human and mouse tissues.                                                                                                          | [Link](YOUR_CAPTAIN_BASE_DOWNLOAD_LINK_HERE)    |
| `CAPTAIN_PBMC`    | Model specifically pretrained on Peripheral Blood Mononuclear Cell (PBMC) data, which is prevalent within the pretraining dataset. | [Link](YOUR_CAPTAIN_PBMC_DOWNLOAD_LINK_HERE)    |
| `CAPTAIN_Human`   | Model specifically pretrained on human single-cell data, which constitutes a significant proportion of the pretraining dataset.                                 | [Link](YOUR_CAPTAIN_HUMAN_DOWNLOAD_LINK_HERE)|

**Gene and Surface Protein Dictionaries:**

Within each model's download folder, you will also find:

* **Gene ID Dictionary:** This dictionary maps gene symbols to numerical IDs, referencing the vocabulary used in [scGPT](https://github.com/bowang-lab/scGPT).
* **Cell Surface Protein Dictionary:** This dictionary maps 387 cell surface protein names to numerical IDs, compiled and curated manually by our team.
