# CAPTAIN: A multimodal foundation model pretrained on co-assayed single-cell RNA and protein
===========================================================================


[![license](https://img.shields.io/badge/python_-3.8.0_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-1.12.0_-blue)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.9.0_-blue)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.8.0_-blue)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/R_-4.2.2_-blue)](https://www.r-project.org/)

Proteins act as the ultimate executors of cellular function, encoding the phenotypic consequences of genomic and transcriptomic programs. Although transcriptomic profiles serve as accessible proxies, they remain incomplete surrogates for the proteomic landscape that ultimately defines cellular phenotypes. Current single-cell foundation models, however, are trained exclusively on transcriptomes, resulting in biased and partial characterizations of cellular states. To address this limitation, we introduce CAPTAIN, a multimodal foundational model pretrained on over four million single cells with concurrently measured transcriptomes and a curated repertoire of 387 surface proteins across diverse human and mouse tissues. Our results show that CAPTAIN learns unified multimodal representations by modeling cross-modality dependencies and capturing the diversity of cellular states across complex biological contexts. CAPTAIN generalizes robustly across both fine-tuning and zero-shot settings, excelling in core downstream tasks such as protein imputation, cell type annotation, and batch harmonization. Beyond improved accuracy in multi-omics integration, CAPTAIN uncovers previously inaccessible mechanisms of protein-mediated intercellular dynamics, including immune interaction patterns linked to COVID-19 severity. CAPTAIN establishes a new paradigm for multimodal single-cell modeling, laying the foundation for comprehensive cellular understanding and virtual cell construction.
