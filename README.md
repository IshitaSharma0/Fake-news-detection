# GA-RoBERTa-FakeNews
 
**A Genetic Algorithm-Based Layer Optimization Framework for Cross-Domain Fake News Detection Using RoBERTa**
 
*Ishita Sharma, Abhishek Bhatt*
*Department of Artificial Intelligence and Robotics / Center for Artificial Intelligence*
*Madhav Institute of Technology and Science, Gwalior, India*
 
---
 
## Overview
 
This repository contains the complete source code for the paper:
 
> **A Genetic Algorithm-Based Layer Optimization Framework for Cross-Domain Fake News Detection Using RoBERTa**
> Submitted to IEEE Latin America Transactions.
 
We propose a Genetic Algorithm (GA)-based framework that automatically selects the optimal subset of RoBERTa transformer layers to fine-tune for cross-domain fake news detection. The framework mitigates negative transfer when adapting a domain-initialized model to heterogeneous target datasets.
 
---
 
## Repository Structure
 
```
GA-RoBERTa-FakeNews/
│
├── README.md
├── requirements.txt
└── notebooks/
    ├── 1_vanilla_isot_roberta.ipynb       # Vanilla RoBERTa and ISOT-initialized baselines
    └── 2_ga_optimized_roberta.ipynb       # GA-based layer optimization (proposed method)
```
 
---
 
## Notebooks
 
### Notebook 1 — Vanilla RoBERTa & ISOT-Initialized Baseline
`notebooks/1_vanilla_isot_roberta.ipynb`
 
This notebook implements and evaluates two baseline models:
- **Vanilla RoBERTa** — standard RoBERTa-base fine-tuned directly on each target dataset
- **ISOT-Initialized RoBERTa** — RoBERTa first fine-tuned on ISOT, then transferred to target datasets
 
---
 
### Notebook 2 — GA-Based Layer Optimization (Proposed Method)
`notebooks/2_ga_optimized_roberta.ipynb`
 
This notebook implements the proposed framework:
- Loads the ISOT-initialized RoBERTa model
- Applies the Genetic Algorithm to search for the optimal layer mask
- Fine-tunes the model using the optimal configuration
- Evaluates on COVID-19, GossipCop, and WELFake datasets
 
---
 
## Datasets
 
The following publicly available datasets are used. Download each from the links below and update the file paths in the notebooks accordingly.
 
 
| Dataset | Domain | Source |
|---|---|---|
| ISOT Fake News | Politics | [Kaggle — rahulogoel/isot-fake-news-dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset) |
| COVID-19 Fake News | Health | [CodaLab — Official CONSTRAINT 2021 dataset](https://competitions.codalab.org/competitions/26655) |
| GossipCop | Entertainment | [FakeNewsNet GitHub](https://github.com/KaiDMML/FakeNewsNet) |
| WELFake | Multi-domain | [Kaggle — saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
 
 


## Reproducibility
 
To ensure reproducible results, all random seeds are fixed at the start of each notebook:
 
```python
import random, numpy as np, torch
 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
 
---
 
## Results
 
| Model | Dataset | Accuracy (%) | Weighted F1 |
|---|---|---|---|
| Vanilla RoBERTa | COVID-19 | 97.52 | 97.50 |
| ISOT Initialized | COVID-19 | 63.50 | 58.00 |
| **Proposed (GA)** | **COVID-19** | **97.19** | **97.35** |
| Vanilla RoBERTa | GossipCop | 86.31 | 86.00 |
| ISOT Initialized | GossipCop | 76.06 | 66.00 |
| **Proposed (GA)** | **GossipCop** | **86.64** | **91.67** |
| Vanilla RoBERTa | WELFake | 99.91 | 100.00 |
| ISOT Initialized | WELFake | 95.11 | 95.00 |
| **Proposed (GA)** | **WELFake** | **99.00** | **99.65** |
 
---
 
## GA Configuration
 
| Parameter | Value |
|---|---|
| Chromosome length | 12 |
| Population size | 5 |
| Generations | 2 |
| Mutation rate | 20% |
| Selection method | Tournament |
| Crossover | Single point |
| Fitness function | Validation F1 score |
 
---
 
## Citation
 
If you use this code in your research, please cite:
 
```
@article{sharma2025garoberta,
  title   = {A Genetic Algorithm-Based Layer Optimization Framework
             for Cross-Domain Fake News Detection Using RoBERTa},
  author  = {Sharma, Ishita and Bhatt, Abhishek},
  journal = {IEEE Latin America Transactions},
  year    = {2025}
}
```
 
---
 
## License
 
This project is intended for academic and research purposes only.
