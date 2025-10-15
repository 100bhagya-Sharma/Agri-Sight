import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm

class BasicPreprocessor:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.train_augmentation = self._create_train_augmentation_pipeline()
        self.test_augmentation = self._create_test_augmentation_pipeline()
    
    def _create_train_augmentation_pipeline(self):
        """Create a basic augmentation pipeline for training data"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ])
    
    def _create_test_augmentation_pipeline(self):
        """Create minimal augmentation pipeline for validation/test data"""
        return A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ])
    
    def preprocess_image(self, image):
        """Preprocess a single image"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is uint8 and in correct range
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentations
            augmented = self.train_augmentation(image=image)
            return augmented['image']
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

class DataPreprocessor:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.train_augmentation = self._create_train_augmentation_pipeline()
        self.test_augmentation = self._create_test_augmentation_pipeline()
    
    def _create_train_augmentation_pipeline(self):
        """Create a balanced augmentation pipeline for training data"""
        return A.Compose([
            # Basic geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Simple color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=1
                ),
            ], p=0.5),
            
            # Mild noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.Blur(blur_limit=3, p=1),
            ], p=0.3),
            
            # Simple affine transformations
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            
            # Always apply preprocessing
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ])
    
    def _create_test_augmentation_pipeline(self):
        """Create minimal augmentation pipeline for validation/test data"""
        return A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ])
    
    def preprocess_image(self, image):
        """Preprocess a single image"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is uint8 and in correct range
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentations
            augmented = self.train_augmentation(image=image)
            return augmented['image']
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def save_image(self, image, output_path):
        """Save the processed image as proper 8-bit RGB"""
        try:
            if image is None:
                return False
            # Ensure image is in [0,255] uint8 before saving
            img = image
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            # Convert back to BGR for OpenCV
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
            return True
        except Exception as e:
            print(f"Error saving image {output_path}: {str(e)}")
            return False
    
    def prepare_dataset(self, data_dir, output_dir, is_training=True):
        """Process entire dataset with augmentations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        total_processed = 0
        failed_images = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Read image with OpenCV
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError(f"Failed to read image: {img_path}")
                    processed_img = self.preprocess_image(image)
                    
                    if processed_img is not None:
                        # Save processed image
                        output_path = os.path.join(output_class_dir, f"proc_{img_name}")
                        if self.save_image(processed_img, output_path):
                            total_processed += 1
                        else:
                            failed_images.append(img_path)
                    else:
                        failed_images.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    failed_images.append(img_path)
        
        # Print processing summary
        print(f"\nProcessing Summary:")
        print(f"Total images processed successfully: {total_processed}")
        print(f"Failed images: {len(failed_images)}")
        if failed_images:
            print("Failed image paths:")
            for path in failed_images:
                print(f"- {path}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Example usage:
    # preprocessor.prepare_dataset("data/raw/color", "data/processed", is_training=True)