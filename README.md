<div align="center">

# Image Caption Generator using Deep Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)](https://matplotlib.org)
[![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge)](https://www.nltk.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)]()

*A deep learning pipeline that generates natural language descriptions for images, combining CNN-based visual feature extraction with LSTM-based language modeling — trained on the Flickr8K dataset.*

</div>

---

A deep learning project that automatically produces descriptive captions for input images. The model encodes visual content using a pretrained InceptionV3 CNN, then decodes it into coherent natural language using an LSTM-based sequence model. The project covers the full pipeline — from raw dataset ingestion and caption preprocessing through model training, optimization, and BLEU-score evaluation.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Background

Automatic image captioning sits at the intersection of computer vision and natural language processing. Given an image, the system must identify salient objects and relationships, then express them in grammatically coherent language — a task that requires tight coupling between visual and linguistic representations.

This project implements the encoder-decoder paradigm: a CNN extracts a fixed-length feature vector from the image, which is fed as the initial context to an LSTM that generates a caption word-by-word. Transfer learning via InceptionV3 (pretrained on ImageNet) is used to obtain rich visual representations without training a CNN from scratch.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [Flickr8K — Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) |
| Images | 8,000 photographs |
| Captions | 5 human-annotated captions per image (40,000 total) |
| Annotations | `captions.txt` — tab-separated image filename and caption |

Ensure both the `Images/` directory and `captions.txt` are placed at the paths expected by the notebook before running.

---

## Project Structure

```
.
├── Image_Caption_Generator_using_Deep_Learning_on_Flickr8K_dataset.ipynb   # Main training and evaluation notebook
└── README.md
```

> **Note:** The Flickr8K dataset (Images/ directory and captions.txt) must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and placed at the paths referenced inside the notebook.

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy pandas matplotlib seaborn plotly tensorflow scikit-learn nltk tqdm pillow
```

After installation, download the required NLTK tokenizer data:

```python
import nltk
nltk.download('punkt')
```

All experiments are self-contained within the Jupyter notebook. No additional configuration is required.

---

## Methodology

The pipeline follows a structured deep learning workflow:

**1. Data Preprocessing**
- Caption loading from `captions.txt` and mapping to image filenames
- Text normalization — lowercasing, punctuation removal, numeric filtering
- Vocabulary construction with frequency thresholding
- Addition of `<start>` and `<end>` sequence tokens

**2. Feature Extraction**
- Images resized to 299×299 and preprocessed for InceptionV3
- InceptionV3 (pretrained, ImageNet weights) used as a frozen feature extractor
- Final classification layer removed; 2048-dimensional pooled features extracted per image
- Features cached to disk to avoid redundant forward passes during training

**3. Sequence Preparation**
- Captions tokenized and integer-encoded using a fitted `Tokenizer`
- Input–output sequence pairs generated via teacher forcing
- Variable-length sequences padded to maximum caption length

**4. Model Training**
- CNN+LSTM model trained on (image feature, partial caption) → next word pairs
- Early stopping on validation loss to prevent overfitting
- Learning rate scheduling for stable convergence

**5. Evaluation**
- Caption quality measured using BLEU-1 and BLEU-4 scores
- Qualitative assessment via side-by-side image and generated caption display

---

## Model Architecture

The model follows an encoder–decoder design:

| Component | Detail |
|-----------|--------|
| Visual Encoder | InceptionV3 (frozen, ImageNet weights) → Dense(256) |
| Caption Embedding | Embedding layer → Dropout → LSTM(256) |
| Decoder | Element-wise addition of visual and language features → Dense(vocab\_size, softmax) |
| Loss | Categorical cross-entropy |
| Optimizer | Adam with learning rate scheduling |

The two branches (image and partial caption) are merged before the output layer, allowing the model to condition each word prediction on both the image content and the words generated so far.

---

## Usage

1. Download the [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the `Images/` folder and `captions.txt` at the paths specified in the notebook.
2. Install dependencies (see [Installation](#installation)).
3. Open `Image_Caption_Generator_using_Deep_Learning_on_Flickr8K_dataset.ipynb` in Jupyter.
4. Run all cells sequentially — feature extraction results are cached after the first run for faster iteration.

---

## Future Work

- **Attention mechanism** — implement Bahdanau attention to allow the decoder to focus on relevant image regions at each step
- **Transformer-based decoder** — replace the LSTM with a Transformer decoder for improved long-range coherence
- **Larger datasets** — scale to MS-COCO (330K images) for improved vocabulary coverage and generalisation
- **Beam search decoding** — replace greedy decoding with beam search for higher-quality caption candidates
- **Deployment** — interactive demo via Streamlit or Gradio allowing users to upload arbitrary images

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Ensure any new code is accompanied by clear inline documentation.

---

## License

This project is released for academic and research purposes. See `LICENSE` for details.
