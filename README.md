# ANTICP3 — Anticancer Protein Prediction

<p align="center">
  <img src="logo.png" alt="ANTICP3 Logo" width="500"/>
</p>

**ANTICP3** is a LLM-based tool for binary classification of proteins into *Anticancer* or *Non-Anticancer* classes, based solely on their primary amino acid sequences. It leverages the powerful [ESM2-t33](https://huggingface.co/facebook/esm2_t33_650M_UR50D) transformer model, fine-tuned specifically for anticancer protein prediction.

> Developed by **Prof. G. P. S. Raghava's Lab**, IIIT-Delhi  
> 📄 Please cite: [ANTICP3](https://webs.iiitd.edu.in/raghava/anticp3)

---

## Features

- Fine-tuned ESM2 model for accurate prediction.
- Accepts input in FASTA format.
- Outputs CSV with predicted labels and probabilities.
- Supports CPU and CUDA for faster inference.
- Easy to integrate into pipelines and large-scale datasets.

---

## Model Details

- **Base Model:** facebook/esm2_t33_650M_UR50D
- **Fine-Tuned On:** Anticancer protein dataset
- **Classification Type:** Binary (Anticancer / Non-Anticancer)
- **Output Format:** CSV with prediction scores and labels

---

## Command-Line Arguments

| Parameter       | Accepted Values            | Description                                                                 |
|-----------------|----------------------------|-----------------------------------------------------------------------------|
| `-i`, `--input` | Path to `.fasta` file      | **(Required)** Input file containing protein sequences in FASTA format.     |
| `-o`, `--output`| Any valid filename (e.g. `results.csv`) | Output CSV file to save predictions. Default is `output.csv`.           |
| `-t`, `--threshold` | Float (0 to 1)             | Classification threshold for deciding Anticancer vs Non-Anticancer. Default is `0.5`. |
| `-d`, `--device`| `cpu` or `cuda`            | Device to run inference on. Defaults to `cpu`. If `cuda` is specified and available, inference runs on GPU. |

## Requirements

You can install the dependencies using either `conda` or `pip`.

### Option 1: Using Conda

Recommended if you're using a Conda environment.

```bash
conda env create -f environment.yaml
conda activate anticp3
```

### Option 2: Using Pip
```bash 
pip install -r requirements.txt
```

## HuggingFace

## 🤗 Inference via Hugging Face

You can also run predictions using the fine-tuned model directly from [Hugging Face Hub](https://huggingface.co/AmishaG/anticp3):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModelForSequenceClassification.from_pretrained("AmishaG/anticp3")

# Example protein sequence
sequence = "MANCVVGYIGERCQYRDLKWWELRGGGGSGGGGSAPAFSVSPASGLSDGQSVSVSVSGAAAGETYYIAQCAPVGGQDACNPATATSFTTDASGAASFSFVVRKSYTGSTPEGTPVGSVDCATAACNLGAGNSGLDLGHVALTFGGGGGSGGGGSDHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCKLEHHHHHH"

# Tokenize and predict
inputs = tokenizer(sequence, return_tensors="pt", truncation=True)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probs, dim=1).item()

labels = {0: "Non-Anticancer", 1: "Anticancer"}
print("Prediction:", labels[prediction])
