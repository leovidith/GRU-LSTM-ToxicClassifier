# Toxic Comment Detection and Classification using GRU, Bi-LSTM, Glove Embeddings
## Overview

The **Multilingual Jigsaw Comment Classification** project tackles the challenge of toxic comment detection using cutting-edge Natural Language Processing (NLP) and deep learning techniques. This solution focuses on multilingual datasets, leveraging pretrained word embeddings and advanced neural architectures for binary classification.

### Key Components

1. **Gated Recurrent Unit (GRU)**:
   - A type of recurrent neural network (RNN) optimized to handle sequential data efficiently.
   - GRU reduces computational complexity by combining the functionalities of the input and forget gates into a single gate.
   - In this project, the GRU model integrates:
     - **Pretrained GloVe embeddings** as input.
     - A **Spatial Dropout layer** to reduce overfitting.
     - A **single dense layer** followed by an **output layer** with a sigmoid activation function.

2. **Bidirectional Long Short-Term Memory (Bi-LSTM)**:
   - An extension of LSTM networks, Bi-LSTMs read input sequences in both forward and backward directions, capturing more context.
   - This architecture is well-suited for understanding multilingual text patterns.
   - The Bi-LSTM model here includes:
     - **Pretrained GloVe embeddings** as input.
     - A **Bidirectional LSTM layer** for robust context learning.
     - A **single dense layer** and an **output layer** with sigmoid activation.

3. **Pretrained GloVe Embeddings**:
   - **GloVe** (Global Vectors for Word Representation) is a popular word embedding model trained on large corpora to represent words as dense vectors.
   - The specific embeddings used are **GloVe 840B.300d**, trained on 840 billion tokens from the Common Crawl dataset, with 300-dimensional vectors.
   - These embeddings ensure semantic similarity between words is captured effectively, aiding model performance.

### Preprocessing and Hyperparameters:
- **Dataset**: Processed to include toxic vs. non-toxic labels only.
- **Maximum Sequence Length**: Set to 1500 for handling long comments.
- **Tokenizer Vocabulary**: Built dynamically from the training data.
- **Batch Size**: 64 samples per iteration for both models.
- **Optimizer**: Adam optimizer used for efficient gradient updates.
- **Loss Function**: Binary cross-entropy to optimize classification.
- **Dropout**: Spatial Dropout rate of 0.3 in both models to prevent overfitting.
- **Training Epochs**: Models trained for 5 epochs to balance training time and performance.

This architecture and parameter design ensure robust and scalable toxic comment classification across multilingual datasets.

## Results

### Key Metrics:
1. **Model 1: GRU**
   - Achieved an **accuracy of 97%** in 5 epochs.
   - Efficiently processed the dataset with minimal computational overhead.

2. **Model 2: Bi-LSTM**
   - Also achieved an **accuracy of 97%** in 5 epochs.
   - However, the Bi-LSTM model required **45 minutes to execute**, highlighting the trade-off between computational complexity and model architecture.

### Insights:
- Both models exhibit comparable accuracy, demonstrating the effectiveness of GloVe embeddings in capturing semantic relationships within the text.
- The GRU model stands out for its computational efficiency, making it more practical for scenarios with time or resource constraints.
- The Bi-LSTM model, though computationally intensive, may provide better context understanding due to its bidirectional processing.

### Visualizations:
- Accuracy and loss plots for both models are included to illustrate training progress and convergence.
<img src="https://github.com/leovidith/GRU-LSTM-ToxicClassifier/blob/main/history.png" width=1000px>

## Agile Features
- **Pretrained Embeddings**: Efficiently utilized GloVe embeddings to initialize word representations, reducing computational overhead.
- **Tokenization & Padding**: Ensured consistent input sizes through robust preprocessing methods.
- **Model Architectures**: Designed scalable models (GRU and Bi-LSTM) optimized for binary classification tasks.
- **Evaluation Pipeline**: Incorporated ROC-AUC metrics for precise model evaluation and comparison.

## Conclusion
The project successfully demonstrates the applicability of pretrained embeddings and neural networks in multilingual toxic comment classification. With high AUC scores, the implemented models are effective tools for real-world moderation systems. Future enhancements could include fine-tuning embeddings and extending datasets for broader language coverage.
