# Vietnamese Tourism VQA System 
This repository contains the source code for a Visual Question Answering (VQA) system specialized in Vietnamese tourism landmarks, using a combination of PhoBERT and Vision Transformer (ViT).
## 📁 Repository Structure

- `configs/`: YAML/JSON config files for training, evaluation, etc.
- `models/`: Model architecture, including PhoBERT + ViT + Co-Attention.
- `scripts/`: Data generation, training, evaluation, and inference scripts.
- `utils/`: Utility functions (e.g., logging, visualization, metrics).

**Note** Due to size constraints, larger folders have been uploaded separately::
- `data/`: Preprocessed images, annotations, tokenizer | https://www.kaggle.com/datasets/dtnaguyen/data-checkpoint-for-vqa
- `checkpoints/`: Trained model weights | https://www.kaggle.com/datasets/dtnaguyen/data-checkpoint-for-vqa

# The directory tree structure should look like this
`checkpoints`
`configs`
`data`
`models`
`script`
`utils`

# Install required packages
pip install -r requirements.txt

# Information
Thesis Title: Hệ thống Visual Question Answering cho các địa danh du lịch Việt Nam
Technologies: PyTorch, HuggingFace Transformers, PhoBERT, Vision Transformer (ViT)
Author: Dat Nguyen
Institution: Van Lang University
Year: 2025

# Contact
Email: nguyenthanhdat241@gmail.com
