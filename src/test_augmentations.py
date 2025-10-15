import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor

def visualize_augmentations(image_path, num_examples=5, save_path=None):
    """
    Visualize the effects of augmentations on a sample image
    Args:
        image_path: Path to the input image
        num_examples: Number of augmented examples to generate
        save_path: Optional path to save the visualization
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Read and convert image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(15, 3))
    
    # Plot original image
    plt.subplot(1, num_examples + 1, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Generate and plot augmented versions
    for i in range(num_examples):
        augmented = preprocessor.train_augmentation(image=image)['image']
        plt.subplot(1, num_examples + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    # Create output directory for visualizations
    output_dir = "augmentation_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test with a few sample images from different classes
    data_dir = "processed_data/train"  # Use processed data directory
    
    for class_name in os.listdir(data_dir)[:3]:  # Test with first 3 classes
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get first image from class
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                save_path = os.path.join(output_dir, f"augmented_{class_name}_{img_name}")
                
                print(f"\nGenerating augmentations for {class_name}")
                visualize_augmentations(img_path, num_examples=5, save_path=save_path)
                break  # Process only first image from each class

if __name__ == "__main__":
    main()