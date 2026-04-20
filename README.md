# PRISM: Perinuclear Ring-based Image Segmentation Method for Acute Lymphoblastic Leukemia Classification

[![Conference](https://img.shields.io/badge/Accepted-SBCAS_2026-success.svg)](https://www.sbcas2026.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-3.10-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Rodrigues Moreira et al. (2026)**

### ⚙️ Pipeline

1. **Preprocessing:** CIELAB color mapping, CLAHE enhancement, and nucleus segmentation.
2. **PRISM Zonal Extraction:** Dynamic generation of proximal and distal concentric zones around the nucleus.
3. **Feature Engineering:** Extraction of multidomain attributes (morphology, chromatic gradients, GLCM, LBP).
4. **Meta-Classification:** Two-level Heterogeneous Ensemble Stacking with Probability Calibration (Platt Scaling).

### 📝 How to Cite

If you use PRISM in your research, please cite the following paper:

```bibtex
@inproceedings{RodriguesMoreira2026,
  title={{PRISM: Perinuclear Ring-based Image Segmentation Method for Acute Lymphoblastic Leukemia Classification}},
  author={Rodrigues Moreira, Larissa Ferreira and Rodrigues, Leonardo Gabriel Ferreira and Moreira, Rodrigo and Backes, Andr{\'e} Ricardo},
  booktitle={Anais do XXVI Simp{\'o}sio Brasileiro de Computa{\c{c}}{\~a}o Aplicada {\`a} Sa{\'u}de (SBCAS 2026)},
  year={2026},
  publisher={SBC},
  address={Brazil}
}
```
