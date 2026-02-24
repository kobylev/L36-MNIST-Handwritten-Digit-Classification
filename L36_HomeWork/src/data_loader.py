import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Test shape: {x_test.shape}, {y_test.shape}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    return (x_train, y_train), (x_test, y_test)

def visualize_samples(x, y, save_path=None):
    idxs = np.random.choice(len(x), 16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[idxs[i]], cmap='gray')
        ax.set_title(f"Label: {y[idxs[i]]}")
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved sample grid to {save_path}")
    plt.close()

def preprocess_data(x, y):
    x = x.astype('float32') / 255.0
    x = x.reshape((x.shape[0], -1))  # Flatten 28x28 -> 784
    y = to_categorical(y, 10)
    return x, y

def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    visualize_samples(x_train, y_train, os.path.join(OUTPUT_DIR, 'mnist_samples.png'))
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)
