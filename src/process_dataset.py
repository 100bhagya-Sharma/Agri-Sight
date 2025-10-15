import os
import json
from data_preprocessing import DataPreprocessor
from tqdm import tqdm

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw', 'color')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process each class
    class_dirs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        print(f"\nProcessing class: {class_name}")
        
        # Create output directory for this class
        output_class_dir = os.path.join(processed_data_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all images in this class
        class_dir = os.path.join(raw_data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image
        for img_name in tqdm(images, desc="Processing images", leave=False):
            input_path = os.path.join(class_dir, img_name)
            output_path = os.path.join(output_class_dir, f"proc_{img_name}")
            
            try:
                # Process the image and save it
                processed_img = preprocessor.preprocess_image(input_path)
                preprocessor.save_image(processed_img, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    main()