# ProCBM  
**Progressive Multi-modal Concept Bottleneck Model**

Official PyTorch implementation of  
**ProCBM: Progressive Multi-modal Concept Bottleneck Model for Interpretable Medical Image Diagnosis**

---


## 🔍 Overview

Concept Bottleneck Models (CBMs) improve interpretability by explicitly reasoning over human-understandable concepts. However, existing CBMs typically rely on **static, unimodal language concepts**, which are often misaligned with complex and diverse medical imaging patterns.

**ProCBM** introduces a **progressive multi-modal concept refinement framework**, where diagnostic concepts are:

- Initialized from language models
- Iteratively refined via interaction with hierarchical visual features
- Dynamically updated through gated multi-modal fusion
- Used as an explicit concept bottleneck for final prediction

This design preserves interpretability while significantly improving diagnostic performance across multiple medical imaging modalities.

---

## ✨ Key Features

- Progressive multi-modal concept refinement  
- Concept-guided visual evidence aggregation  
- Gated fusion of language priors and visual features  
- Label-free concept learning from language models  
- Explicit concept bottleneck for transparent decision making  

---
##  Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1vf6X44zALelFXQNCAmg0_VizT4yxRkse?usp=drive_link)

---

## 🛠 Installation

```bash
conda create -n procbm python=3.9 -y
conda activate procbm
pip install -r requirements.txt
```

---

## ▶️ Running ProCBM (ISIC2018 Example) 


```bash
# isic2018
# cm
# busi
# miniddsm
# nct
# siim
# idrid

python train_procbm.py \
  -d isic2018 \
  --data-path ./dataset/isic2018/ \
  --gpu 0 \
  -e 150 \
  -w 0.001
```

---

## 📄 License

For academic research use only.
