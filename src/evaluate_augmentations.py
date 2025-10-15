import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
import cv2
from model import DiseaseClassifier
from data_preprocessing import DataPreprocessor, BasicPreprocessor

def train_and_evaluate(data_dir, preprocessor, input_shape=(224, 224, 3), batch_size=32, epochs=15):
    """Evaluate a pre-trained model using the given preprocessor for validation data."""
    # Create data generators
    train_datagen = ImageDataGenerator(
        validation_split=0.2
        # Removed rescaling since normalization is handled by preprocessors
    )

    # Load and prepare the data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_datagen = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=preprocessor.preprocess_image
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Initialize the model
    num_classes = len(train_generator.class_indices)
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = DiseaseClassifier(input_shape=input_shape, num_classes=num_classes, checkpoint_dir=checkpoint_dir)
    
    print(f"\nLoading pre-trained model for {preprocessor.__class__.__name__} evaluation...")
    model.load(os.path.join(checkpoint_dir, 'best_model.h5'))
    
    # Evaluate the model
    print("\nEvaluating model with preprocessor augmentations on validation data...")
    report, conf_matrix = model.evaluate(validation_generator)
    
    return None, report, conf_matrix

def plot_training_history(basic_history, advanced_history):
    """Plot training histories for comparison."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(basic_history.history['accuracy'], label='Basic - Training')
    plt.plot(basic_history.history['val_accuracy'], label='Basic - Validation')
    plt.plot(advanced_history.history['accuracy'], label='Advanced - Training')
    plt.plot(advanced_history.history['val_accuracy'], label='Advanced - Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(basic_history.history['loss'], label='Basic - Training')
    plt.plot(basic_history.history['val_loss'], label='Basic - Validation')
    plt.plot(advanced_history.history['loss'], label='Advanced - Training')
    plt.plot(advanced_history.history['val_loss'], label='Advanced - Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/augmentation_comparison.png')
    plt.close()

def main():
    # Set up data directory
    data_dir = os.path.join('processed_data', 'train')
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} not found!")
    
    print("Starting augmentation evaluation experiment without retraining...")
    print(f"Using data directory: {data_dir}")
    print(f"Number of classes: {len(os.listdir(data_dir))}")
    
    # Create preprocessors
    basic_preprocessor = BasicPreprocessor()
    advanced_preprocessor = DataPreprocessor()
    
    # Evaluate with basic augmentations on validation
    print("\nEvaluating with basic augmentations...")
    _, basic_report, basic_conf_matrix = train_and_evaluate(data_dir, basic_preprocessor)
    
    # Evaluate with advanced augmentations on validation
    print("\nEvaluating with advanced augmentations...")
    _, advanced_report, advanced_conf_matrix = train_and_evaluate(data_dir, advanced_preprocessor)
    
    # Save evaluation reports (no plot since no training)
    os.makedirs('results', exist_ok=True)
    with open('results/basic_augmentation_report.txt', 'w') as f:
        f.write("Basic Augmentation Results:\n")
        f.write(basic_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(basic_conf_matrix))
    
    with open('results/advanced_augmentation_report.txt', 'w') as f:
        f.write("Advanced Augmentation Results:\n")
        f.write(advanced_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(advanced_conf_matrix))
    
    print("\nEvaluation complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()