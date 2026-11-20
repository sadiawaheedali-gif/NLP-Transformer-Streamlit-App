# Multilingual NLP Streamlit Application

This project is a user-friendly Streamlit application that integrates two fine-tuned Hugging Face Transformer models.

## Models

### 1. mBERT - Natural Language Inference

The first model is a fine-tuned multilingual BERT model based on:

`google-bert/bert-base-multilingual-cased`

It was fine-tuned on the XNLI dataset for Natural Language Inference.

The application accepts:

- Premise
- Hypothesis

and predicts:

- Entailment
- Neutral
- Contradiction

### 2. GPT-2 - Balochi Text Generation

The second model is a fine-tuned GPT-2 model based on:

`gpt2`

It was fine-tuned using a Balochi instruction-response dataset.

The application accepts an instruction and generates a text response.

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Hugging Face Hub

## Application Features

- Select between two fine-tuned models
- Perform multilingual NLI
- Generate Balochi text
- Simple user interface
- Models are loaded directly from Hugging Face

## Hugging Face Models

mBERT model:

`Soho77/mbert-xnli-results`

GPT-2 model:

`Soho77/gpt2-balochi-results`

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
