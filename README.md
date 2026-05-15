# IMDB Sentiment Analysis

Simple neural network for sentiment analysis using GloVe embeddings.

## Model
- Bidirectional LSTM with pre-trained GloVe embeddings (frozen)
- Dataset: IMDB reviews
- Vocab: 4993 words from tiny_glove.json

## CI/CD
GitHub Actions automatically trains and uploads model to Hugging Face Hub on every push to main.
