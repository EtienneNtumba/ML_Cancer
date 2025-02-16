import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, LSTM, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATASET_PATH = config['data']['dataset_path']
EPOCHS = config['train']['epochs']
BATCH_SIZE = config['train']['batch_size']
LEARNING_RATE = config['train']['learning_rate']
MODEL_TYPE = config['train']['model_type']
SAVE_PATH = config['train']['save_model_path']

def load_data():
    """Load and preprocess gene expression dataset."""
    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, 1:].values  # Gene expression values
    y = df.iloc[:, 0].values  # Labels
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def build_cnn_model(input_shape):
    """Create a CNN model for gene expression analysis."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_rnn_model(input_shape):
    """Create an RNN model for gene expression analysis."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_shape, 1)),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_keras_model():
    """Train the selected deep learning model."""
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]  # Reshape for CNN/RNN
    
    if MODEL_TYPE == "cnn":
        model = build_cnn_model(X_train.shape[1])
    elif MODEL_TYPE == "rnn":
        model = build_rnn_model(X_train.shape[1])
    else:
        raise ValueError("Invalid model type. Choose 'cnn' or 'rnn'.")
    
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    model.save(os.path.join(SAVE_PATH, "gene_expression_model.keras"))
    return history


def visualize_results(history):
    """Plot training accuracy and loss curves."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

if __name__ == "__main__":
    history = train_keras_model()
    visualize_results(history)
    print("Training complete.")
