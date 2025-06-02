# Image Caption Generator using Deep Learning on Flickr8K Dataset

This repository contains a deep learning project that generates natural language captions for images using the Flickr8K dataset. The model leverages convolutional neural networks (CNNs) for feature extraction and recurrent neural networks (RNNs) for language modeling.

## Overview

The goal of this project is to train a model that can generate accurate and descriptive captions for input images. The notebook covers the following steps:

- Dataset loading and preprocessing
- Caption cleaning and tokenization
- Feature extraction from images using InceptionV3
- Sequence padding and vectorization
- Model architecture using CNN + LSTM
- Training with early stopping and learning rate scheduling
- Evaluation using BLEU score

## Dataset

The project uses the [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which contains:

- 8,000 images
- 5 captions per image
- Annotations in a text file (`captions.txt`)

Ensure that both the `Images` directory and `captions.txt` file are available in the specified paths.

## Requirements

To run this project, you need the following Python libraries:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- plotly  
- tensorflow  
- sklearn  
- nltk  
- tqdm  
- PIL

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn plotly tensorflow scikit-learn nltk tqdm pillow
 
