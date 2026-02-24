from src.data_loader import load_and_prepare_data
from src.model import build_model, build_improved_model
from src.training import train_model, plot_training_history, compare_models
from src.evaluation import predict_single, analyze_errors, plot_confusion_matrix
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    # Build and train baseline model
    print("\n--- Baseline Model ---")
    model = build_model()
    history = train_model(model, x_train, y_train)
    plot_training_history(history, os.path.join(OUTPUT_DIR, 'training_history.png'))

    # Evaluate baseline model
    print("\n--- Evaluation ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    predict_single(model, x_test, y_test, digit=7, save_path=os.path.join(OUTPUT_DIR, 'prediction.png'))
    analyze_errors(model, x_test, y_test, save_path=os.path.join(OUTPUT_DIR, 'misclassified.png'))
    plot_confusion_matrix(model, x_test, y_test, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    # Train and compare improved model
    print("\n--- Improved Model ---")
    improved_model = build_improved_model()
    improved_history = train_model(improved_model, x_train, y_train)
    improved_loss, improved_acc = improved_model.evaluate(x_test, y_test, verbose=2)
    print(f"Improved model test accuracy: {improved_acc*100:.2f}%")

    # Compare both models
    print("\n--- Model Comparison ---")
    compare_models(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
