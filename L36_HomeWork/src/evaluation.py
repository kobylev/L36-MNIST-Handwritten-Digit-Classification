import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_single(model, x, y, digit=7, save_path=None):
    idxs = np.where(np.argmax(y, axis=1) == digit)[0]
    idx = np.random.choice(idxs)
    img = x[idx].reshape(28, 28)
    pred = model.predict(x[idx:idx+1])[0]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {digit}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(range(10), pred)
    plt.title("Softmax Probabilities")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.xticks(range(10))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved prediction to {save_path}")
    plt.close()
    return np.argmax(pred)

def analyze_errors(model, x, y_true, save_path=None):
    y_pred = np.argmax(model.predict(x), axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    errors = np.where(y_pred != y_true_labels)[0]
    print(f"Total misclassified: {len(errors)}")
    idxs = np.random.choice(errors, min(25, len(errors)), replace=False)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(idxs):
        plt.subplot(5, 5, i+1)
        img = x[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {y_pred[idx]}", color='red')
        plt.xlabel(f"True: {y_true_labels[idx]}", color='blue')
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved misclassified grid to {save_path}")
    plt.close()

def plot_confusion_matrix(model, x, y_true, save_path=None):
    y_pred = np.argmax(model.predict(x), axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()
    # Print top confused pairs
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(cm2.ravel())[::-1], (10, 10)))[0]
    print("Top confused digit pairs:")
    for i in range(5):
        a, b = pairs[i]
        if cm2[a, b] > 0:
            print(f"{a} vs {b}: {cm2[a, b]} times")
