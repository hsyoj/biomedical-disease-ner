# Biomedical Disease NER on the NCBI Disease Corpus

Fine-tuning and comparing four biomedical/clinical BERT models for **disease named entity recognition (NER)** on the [NCBI Disease corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/). The project covers the full pipeline: parsing the PubTator-format corpus, aligning character-level annotations to subword tokens, fine-tuning token classifiers (with an optional CRF layer), and evaluating with entity-level precision, recall and F1.

## Overview

| Step | Script | What it does |
|------|--------|--------------|
| 1. Preprocessing | `DataProcess.py` | Parses the NCBI train/dev/test files, builds character-level BIO labels, maps them to each model's subword tokens via offset mapping, and writes CoNLL files. |
| 2. Training & evaluation | `Training.py` | Fine-tunes each model for token classification, evaluates on the test set with `seqeval`, saves the best checkpoint and plots training curves. |

### Models compared

| Key | Model |
|-----|-------|
| `Biomed` | [microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract) |
| `Sap` | [cambridgeltl/SapBERT-from-PubMedBERT-fulltext](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) |
| `Bio` | [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) |
| `Clinical` | [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) |

If a local folder with the same name exists (e.g. `./BioBERT`), it is used instead of downloading from the Hugging Face Hub.

### Entity labels

The corpus annotates four disease mention types, giving 9 BIO labels (`O` plus `B-`/`I-` for each type):

- `SpecificDisease`
- `DiseaseClass`
- `Modifier`
- `CompositeMention`

## Features

- **Character-to-token label alignment**: annotations are first expanded to per-character BIO tags, then assigned to subword tokens using tokenizer offset mappings, so every tokenizer gets correctly aligned labels.
- **Multiple backbones** with a shared, reusable training class (`BertTrainer`).
- **Optional CRF layer** (`pytorch-crf`) on top of the encoder (`crf=True`).
- **Optional layer freezing** of the embeddings and bottom N encoder layers.
- **Configurable subword labeling**: label all subwords, or only the first subword of each word.
- **Early stopping** (patience 3) and best-model selection by validation F1.
- **Entity-level metrics** (precision / recall / F1) via `seqeval`, plus token accuracy.
- Loss, F1 and accuracy curves plotted after training.

## Project Structure

```
.
├── DataProcess.py            # PubTator -> CoNLL conversion
├── Training.py               # Fine-tuning and evaluation
├── data/                     # Input corpus (you provide)
│   ├── NCBItrainset_corpus.txt
│   ├── NCBIdevelopset_corpus.txt
│   └── NCBItestset_corpus.txt
├── processed/                # Generated CoNLL files
├── results/                  # Trainer checkpoints (generated)
└── saved_models/             # Final models, one folder per backbone (generated)
```

## Installation

Python 3.10+ is required.

```bash
pip install torch transformers datasets evaluate seqeval pytorch-crf matplotlib
```

## Usage

**1. Get the data.** Download the NCBI Disease corpus and place the three PubTator-format files in `data/`:

- `NCBItrainset_corpus.txt`
- `NCBIdevelopset_corpus.txt`
- `NCBItestset_corpus.txt`

**2. Convert to CoNLL format.**

```bash
mkdir processed
python DataProcess.py
```

This creates `processed/NCBI_{train,dev,test}_{Biomed,Sap,Bio,Clinical}.conll`.

**3. Train and evaluate.**

```bash
python Training.py
```

By default this trains all four models in turn. The final model for each is saved to `saved_models/<Biomed|Sap|Bio|Clinical>`.

### Training a single model

```python
from Training import BertTrainer

bt = BertTrainer(
    model_path="dmis-lab/biobert-base-cased-v1.2",
    data_suffix="Bio",
    bs=16,
    lr=3e-5,
    eps=40,
    label_all_tokens=False,
    freeze=0,      # number of bottom encoder layers to freeze
    crf=False,     # set True to add a CRF layer
)
bt.train_eval()
```

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Learning rate | 3e-5 (linear schedule, 10% warmup) |
| Max epochs | 40 (early stopping, patience 3) |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Best-model metric | validation F1 |

## Results

Fill in after running the experiments (test set, entity-level):

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| BiomedBERT-large | – | – | – |
| SapBERT | – | – | – |
| BioBERT | – | – | – |
| Bio_ClinicalBERT | – | – | – |

## Notes

- Trained model weights are large and are not stored in this repository. Add `saved_models/`, `results/` and `*.pkl` to `.gitignore`, and host weights on Hugging Face Hub or GitHub Releases if you want to share them.
- `Biomed` and `Sap` are uncased models, so they are tokenized with `do_lower_case=True`.
- `fp16` is commented out in `Training.py`; enable it if your GPU supports mixed precision.

## Acknowledgements

- NCBI Disease corpus: Doğan, R. I., Leaman, R., & Lu, Z. (2014). *NCBI disease corpus: A resource for disease name recognition and concept normalization.* Journal of Biomedical Informatics.
- Pretrained models from Microsoft, Cambridge LTL, DMIS Lab (Korea University) and Emily Alsentzer et al.
