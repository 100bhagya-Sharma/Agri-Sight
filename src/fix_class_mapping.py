import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'processed_data', 'train')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MAPPING_PATH = os.path.join(MODELS_DIR, 'class_mapping.json')
IMG_SIZE = 128

if __name__ == '__main__':
    if not os.path.isdir(TRAIN_DIR):
        raise SystemExit(f'Train directory not found: {TRAIN_DIR}')

    print(f'Building class_indices from: {TRAIN_DIR}')
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    # Invert mapping to index -> class_name
    idx_to_class = {int(idx): str(cls) for cls, idx in gen.class_indices.items()}
    idx_to_class_sorted = {str(k): idx_to_class[k] for k in sorted(idx_to_class.keys())}

    print('Class mapping (first 10 entries):')
    for k in sorted(idx_to_class.keys())[:10]:
        print(f'  {k}: {idx_to_class[k]}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MAPPING_PATH, 'w') as f:
        json.dump(idx_to_class_sorted, f, indent=4)
    print(f'Wrote mapping to: {MAPPING_PATH}')