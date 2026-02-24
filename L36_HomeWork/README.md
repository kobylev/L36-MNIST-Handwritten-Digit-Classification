
# 🔢 MNIST Handwritten Digit Classification

A comprehensive deep learning project for classifying handwritten digits using Keras/TensorFlow. This project demonstrates the full pipeline: loading data, building and training neural networks, evaluating results, and visualizing outputs. The notebook and code are designed for both educational and practical use.

---

## 📁 Repository Structure

```
L36_HomeWork/
├── main.py                                # Main entry point
├── src/
│   ├── __init__.py                        # Package init
│   ├── data_loader.py                     # Data loading & preprocessing
│   ├── model.py                           # Neural network architecture
│   ├── training.py                        # Training & visualization
│   └── evaluation.py                      # Prediction & error analysis
├── output/                                # Generated visualizations
│   ├── mnist_samples.png                  # Sample digit images
│   ├── training_history.png               # Loss/accuracy curves
│   ├── prediction.png                     # Single prediction example
│   ├── misclassified.png                  # Error analysis grid
│   └── confusion_matrix.png               # 10x10 confusion matrix
├── MNIST_Classification_Notebook.ipynb    # Jupyter Notebook (for Google Colab)
├── README.md                              # This documentation
└── .gitignore                             # Git ignore rules
```

### Module Overview

| File                | ~Lines | Purpose                                 |
|---------------------|--------|-----------------------------------------|
| main.py             | ~80    | Orchestrates the complete pipeline      |
| data_loader.py      | ~75    | Load MNIST, visualize samples, preprocess |
| model.py            | ~75    | Build & compile neural networks         |
| training.py         | ~85    | Train models, plot history, compare     |
| evaluation.py       | ~95    | Predictions, error analysis, confusion matrix |

All Python files are under 150 lines ✓

---

## 🚀 How to Run

### Option 1: Local Execution (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn

# Run the complete pipeline
python main.py
# Output: All visualizations are saved to the `output/` folder.
```

### Option 2: Google Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `MNIST_Classification_Notebook.ipynb`
3. Click Runtime → Run all

> Tip: Enable GPU: Runtime → Change runtime type → GPU

---

## 🔄 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────┘

	┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
	│  MNIST Data  │ ──▶ │  Preprocessing   │ ──▶ │  Neural Network  │
	│  (Raw Input) │     │  (data_loader)   │     │    (model.py)    │
	└──────────────┘     └──────────────────┘     └──────────────────┘
				│                      │                        │
				▼                      ▼                        ▼
	60,000 train images    • Normalize (0-1)      • 784 input neurons
	10,000 test images     • One-hot encode       • 3 hidden layers
	28×28 grayscale        • Flatten to 784D      • 10 output neurons
																												│
												 ┌──────────────────────────────┘
												 ▼
				 ┌───────────────────────────────────────────────────┐
				 │                  TRAINING (training.py)           │
				 │  • Epochs: 20  • Batch: 128  • Val Split: 10%     │
				 └───────────────────────────────────────────────────┘
												 │
												 ▼
				 ┌───────────────────────────────────────────────────┐
				 │              EVALUATION (evaluation.py)           │
				 │  • Predictions  • Error Analysis  • Confusion Mx  │
				 └───────────────────────────────────────────────────┘
```

---

## 📊 Output Visualizations

### 1. MNIST Sample Images
Random samples from the training dataset showing handwriting variability.

![MNIST Samples](https://github.com/alienspirit7/L36_HomeWork/raw/main/output/mnist_samples.png)

### 2. Training History
Loss and accuracy curves showing model learning over 20 epochs.

![Training History](https://github.com/alienspirit7/L36_HomeWork/raw/main/output/training_history.png)

### 3. Single Prediction Example
Model prediction on a specific digit with confidence scores.

![Prediction](https://github.com/alienspirit7/L36_HomeWork/raw/main/output/prediction.png)

### 4. Misclassified Images
Grid of incorrectly classified samples (Red = Predicted, Blue = Actual).

![Misclassified](https://github.com/alienspirit7/L36_HomeWork/raw/main/output/misclassified.png)

### 5. Confusion Matrix
10×10 matrix showing prediction patterns. Diagonal = correct predictions.

![Confusion Matrix](https://github.com/alienspirit7/L36_HomeWork/raw/main/output/confusion_matrix.png)

---

## 📖 Step-by-Step Explanation

### Section 1: Data Loading & Exploration
Module: `src/data_loader.py` → `load_mnist()`, `visualize_samples()`

• Loads the MNIST dataset (built into Keras)
• Displays dataset statistics
• Visualizes 16 random samples in a 4×4 grid

### Section 2: Preprocessing
Module: `src/data_loader.py` → `preprocess_data()`

| Step           | From      | To        | Purpose                  |
|----------------|-----------|-----------|--------------------------|
| Normalization  | [0-255]   | [0-1]     | Faster convergence       |
| One-Hot Encoding | int (0-9) | 10D vector | For crossentropy loss    |
| Flattening     | 28×28     | 784D      | Dense layer input        |

### Section 3: Model Architecture
Module: `src/model.py` → `build_model()`

```
Input(784) → Dense(512,ReLU) → Dense(256,ReLU) → Dense(128,ReLU) → Output(10,Softmax)
```

• Dropout (0.2): Prevents overfitting
• ReLU: Efficient, avoids vanishing gradients
• Softmax: Outputs probability distribution

### Section 4: Compilation
Module: `src/model.py` → `compile_model()`

| Step     | Setting                  | Purpose                      |
|----------|--------------------------|------------------------------|
| Loss     | Categorical Crossentropy | Multi-class classification   |
| Optimizer| Adam (lr=0.001)          | Adaptive learning rate       |
| Metric   | Accuracy                 | Intuitive performance measure|

### Section 5: Training & Monitoring
Module: `src/training.py` → `train_model()`, `plot_training_history()`

• Epochs: 20 | Batch: 128 | Validation: 10%
• Generates loss and accuracy curves
• Detects overfitting via validation gap

### Section 6: Prediction
Module: `src/evaluation.py` → `predict_single()`

• Selects a specific digit (e.g., '7') from test set
• Shows probability distribution for all classes
• Visualizes image + bar chart

### Section 7: Error Analysis
Module: `src/evaluation.py` → `analyze_errors()`, `plot_confusion_matrix()`

• Identifies all misclassified images
• Displays 25 error examples
• Generates 10×10 confusion matrix
• Reports top confused digit pairs

### Section 8: Hyperparameter Optimization
Module: `src/model.py` → `build_improved_model()`

| Param         | Baseline | Improved |
|---------------|----------|----------|
| Learning Rate | 0.001    | 0.0005   |
| Hidden Layers | 3        | 4        |
| Dropout       | 0.2      | 0.3      |
| Epochs        | 20       | 25       |

---

## 🎯 Results

| Model    | Accuracy | Loss   |
|----------|----------|--------|
| Original | 98.29%   | 0.0771 |
| Improved | 98.38%   | 0.0731 |

Improvement: +0.09% accuracy

---

## 🔧 Troubleshooting

- **Slow training?** Use a subset of data (edit `main.py`):
	```python
	X_train_p = X_train_p[:10000]
	y_train_p = y_train_p[:10000]
	```
- **Memory issues?** Reduce batch size in `train_model()`:
	```python
	train_model(model, X_train_p, y_train_p, batch_size=32)
	```
- **TensorFlow not found?** Install with `pip install tensorflow`.
- **Plots not saving?** Ensure `output/` exists and is writable.
- **Low accuracy?** Check preprocessing, model structure, and training parameters.
- **Colab GPU:** In Colab, select `Runtime > Change runtime type > GPU` for faster training.

---

## 📝 License

Educational project for learning purposes.

---

## Additional Links
- [Project Repository](https://github.com/alienspirit7/L36_HomeWork)
- [Issues](https://github.com/alienspirit7/L36_HomeWork/issues)
- [Pull requests](https://github.com/alienspirit7/L36_HomeWork/pulls)
- [Google Colab Notebook](https://colab.research.google.com/)