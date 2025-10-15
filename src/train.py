import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
from model import DiseaseClassifier
import argparse


def load_class_mapping(train_dir):
    """Load class names from training directory."""
    print("\nLoading class mapping from training directory...")
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    
    class_names = sorted(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
    return class_names


def create_data_generators(data_dir, img_size, batch_size):
    """Create train, validation, and test data generators."""
    print("\nSetting up data generators...")
    
    # Verify directory structure
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Just rescale to [0,1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    print("Creating training generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    print("\nCreating validation generator...")
    valid_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\nCreating test generator...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator


def parse_args():
    parser = argparse.ArgumentParser(description='Train plant disease classifier')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset root containing train/val/test (default: processed_data)')
    parser.add_argument('--img_size', type=int, default=128, help='Input image size (default: 128)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--fine_tune_epochs', type=int, default=30, help='Fine-tune epochs (default: 30)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--output_model_path', type=str, default=None, help='Path to save final model (.h5)')
    parser.add_argument('--output_mapping_path', type=str, default=None, help='Path to save class mapping (.json)')
    return parser.parse_args()


def main():
    print("Starting plant disease classification training pipeline...")
    
    args = parse_args()
    
    # Configuration
    img_size = args.img_size  # Reduced size for faster training with custom CNN
    batch_size = args.batch_size
    epochs = args.epochs
    fine_tune_epochs = args.fine_tune_epochs
    
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(base_dir, 'processed_data')
    data_dir = args.data_dir or default_data_dir
    models_dir = os.path.join(base_dir, 'models')
    default_checkpoint_dir = os.path.join(models_dir, 'checkpoints')
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir
    
    # Output paths
    default_model_path = os.path.join(models_dir, 'final_model.h5')
    default_mapping_path = os.path.join(models_dir, 'class_mapping.json')
    output_model_path = args.output_model_path or default_model_path
    output_mapping_path = args.output_mapping_path or default_mapping_path
    
    # Create necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_dir}")
    print(f"Final model will be saved to: {output_model_path}")
    print(f"Class mapping will be saved to: {output_mapping_path}")
    
    # Get class names and create data generators
    class_names = load_class_mapping(os.path.join(data_dir, 'train'))
    train_generator, valid_generator, test_generator = create_data_generators(
        data_dir, img_size, batch_size
    )
    
    # Initialize and build model
    print("\nInitializing model...")
    model = DiseaseClassifier(
        input_shape=(img_size, img_size, 3),  # RGB input
        num_classes=len(class_names),
        checkpoint_dir=checkpoint_dir
    )
    model.build_model()
    print("Model built successfully!")
    
    # Initial training phase
    print("\nStarting training phase...")
    history = model.train(train_generator, valid_generator, epochs=epochs)
    print("Initial training completed!")
    
    # Fine-tuning phase with lower learning rate
    print("\nStarting fine-tuning phase...")
    fine_tune_history = model.fine_tune(train_generator, valid_generator, epochs=fine_tune_epochs)
    print("Fine-tuning completed!")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    report, conf_matrix = model.evaluate(test_generator)
    print("\nClassification Report:")
    print(report)
    
    # Save final model and class mapping
    print("\nSaving model and class mapping...")
    model.save_model(
        output_model_path,
        output_mapping_path,
        test_generator
    )
    print("Training pipeline completed successfully!")


if __name__ == '__main__':
    main()