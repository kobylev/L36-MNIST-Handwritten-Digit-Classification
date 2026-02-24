import matplotlib.pyplot as plt
import numpy as np
from .model import build_model, build_improved_model

def train_model(model, x_train, y_train, epochs=20, batch_size=128, val_split=0.1):
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=2
    )
    return history

def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training history to {save_path}")
    plt.close()

def compare_models(x_train, y_train, x_test, y_test):
    print("Training baseline model...")
    model1 = build_model()
    hist1 = train_model(model1, x_train, y_train)
    score1 = model1.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline model test accuracy: {score1[1]*100:.2f}%")

    print("Training improved model...")
    model2 = build_improved_model()
    hist2 = train_model(model2, x_train, y_train)
    score2 = model2.evaluate(x_test, y_test, verbose=0)
    print(f"Improved model test accuracy: {score2[1]*100:.2f}%")

    print("\nResults Table:")
    print("| Model     | Test Accuracy |")
    print("|-----------|--------------|")
    print(f"| Baseline  | {score1[1]*100:.2f}%      |")
    print(f"| Improved  | {score2[1]*100:.2f}%      |")
    return (score1, score2)
