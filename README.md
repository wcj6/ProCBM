# ProCBM: Progressive Multi-modal Concept Bottleneck Model

Official PyTorch implementation of **ProCBM: Progressive Multi-modal Concept Bottleneck Model for Interpretable Medical Image Diagnosis**

📄 IEEE Transactions on Medical Imaging (under review)  
🔗 Project Page / Paper: *to be updated*

> ⚠️ **Code Release Notice**  
> The codebase is currently being cleaned and organized for public release.  
> We will release the full implementation, training scripts, and pretrained models as soon as possible.  
> Please stay tuned.


---

## 🔍 Overview

Concept Bottleneck Models (CBMs) provide interpretability by explicitly reasoning over human-understandable concepts. However, existing CBMs rely on **static, unimodal language concepts**, which are often misaligned with complex medical visual patterns.

**ProCBM** formulates concept prediction as a **progressive multi-modal querying process**:

- Each diagnostic concept is initialized from language
- Iteratively refined via interaction with visual features
- Produces dynamic, context-aware multi-modal concept queries
- Maintains explicit interpretability while improving diagnostic performance

---

## ✨ Key Contributions

- **Progressive Multi-modal Refinement (PMR)**  
  Iteratively refines language-derived concepts into multi-modal concept queries.

- **Cross-modal Distillation (CMD)**  
  Extracts concept-specific visual evidence from multi-scale visual features.

- **Gated Multi-modal Fusion (GMF)**  
  Dynamically balances prior concept knowledge and visual evidence.

- **Label-free Concept Learning**  
  Concepts are generated via large language models (LLMs), without dense concept annotations.

- **MLLM-based Faithfulness Evaluation**  
  Uses an external multimodal LLM to assess whether activated concepts are sufficient to justify predictions.

---

## 🧠 Method Summary

Given an image \( x \) and a set of diagnostic concepts \( \{c_i\}_{i=1}^k \):

1. Encode concepts using a frozen text encoder (e.g., BiomedCLIP)
2. Extract hierarchical visual features from a vision backbone
3. Iteratively apply **PMR blocks**:
   - CMD: concept-guided visual evidence aggregation
   - GMF: gated update of concept queries
4. Project refined concept queries into a **concept bottleneck layer**
5. Perform final classification based on concept activations

---

## 📊 Benchmarks

ProCBM is evaluated on **7 medical imaging datasets** covering diverse modalities:

| Dataset | Modality | Task |
|-------|--------|------|
| ISIC2018 | Dermoscopy | Skin lesion classification |
| IDRiD | Fundus | Diabetic retinopathy grading |
| BUSI | Ultrasound | Breast tumor classification |
| MiniDDSM | X-ray | Mammography |
| SIIM | X-ray | Pneumothorax |
| CM | X-ray | Cardiomegaly |
| NCT-CRC-HE | Histopathology | Tissue classification |

ProCBM consistently achieves **state-of-the-art accuracy and balanced accuracy**, outperforming prior CBMs and black-box models.

---

## 🛠 Installation

### Environment

```bash
conda create -n procbm python=3.9 -y
conda activate procbm
pip install -r requirements.txt
