import os
import json
import random
from glob import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='Quick debug evaluation for plant disease model')
    parser.add_argument('--model_path', type=str, default=os.path.join(BASE_DIR, 'models', 'final_model.h5'))
    parser.add_argument('--mapping_path', type=str, default=os.path.join(BASE_DIR, 'models', 'class_mapping.json'))
    parser.add_argument('--test_dir', type=str, default=os.path.join(BASE_DIR, 'processed_data', 'test'))
    parser.add_argument('--classes', type=str, nargs='*', default=None,
                        help='Optional list of class folder names to sample from (space-separated). If omitted, use defaults.')
    parser.add_argument('--per_class', type=int, default=2, help='Number of images to sample per class (default: 2)')
    return parser.parse_args()


DEFAULT_SAMPLE_CLASSES = [
    'Apple___Black_rot',
    'Potato___Late_blight',
    'Tomato___Leaf_Mold',
    'Blueberry___healthy',
    'Corn_(maize)___Common_rust_',
    'Tomato___healthy',
]

random.seed(42)
np.set_printoptions(suppress=True, precision=4)


def load_model_and_mapping(model_path, mapping_path):
    print('Loading model...')
    model = tf.keras.models.load_model(model_path)
    print('Model loaded.')
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)  # str(index) -> class_name
    return model, class_mapping


def determine_input_size(model):
    input_shape = getattr(model, 'input_shape', None)
    if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
        H, W = int(input_shape[1]), int(input_shape[2])
    else:
        H, W = 128, 128
    print(f'Model input size: {H}x{W}')
    return H, W


def preprocess(img_path, W, H):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((W, H))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0), arr


def topk(probs, k=3):
    idx = np.argsort(probs)[-k:][::-1]
    return [(int(i), float(probs[i])) for i in idx]


def evaluate_one(model, class_mapping, img_path, W, H):
    x, arr = preprocess(img_path, W, H)
    preds = model.predict(x, verbose=0)[0]
    if not np.isclose(preds.sum(), 1.0, atol=1e-3):
        preds = tf.nn.softmax(preds).numpy()
    k3 = topk(preds, 3)
    print(f"- Image: {os.path.relpath(img_path, BASE_DIR)}")
    print(f"  input min/mean/max: {arr.min():.4f}/{arr.mean():.4f}/{arr.max():.4f}")
    for i, p in k3:
        label = class_mapping.get(str(i), str(i))
        print(f"  top: {label:45s} -> {p*100:6.2f}% (idx={i})")
    best_idx = int(np.argmax(preds))
    best_label = class_mapping.get(str(best_idx), str(best_idx))
    print(f"  => predicted: {best_label} ({preds[best_idx]*100:.2f}%)\n")


def collect_samples(test_dir, target_classes, per_class):
    samples = []
    used_classes = []
    for cls in target_classes:
        folder = os.path.join(test_dir, cls)
        if not os.path.isdir(folder):
            continue
        files = glob(os.path.join(folder, '*.jpg')) + glob(os.path.join(folder, '*.png')) + glob(os.path.join(folder, '*.jpeg'))
        if not files:
            continue
        random.shuffle(files)
        used_classes.append(cls)
        samples.extend(files[:per_class])
    return used_classes, samples


def main():
    args = parse_args()

    model, class_mapping = load_model_and_mapping(args.model_path, args.mapping_path)
    H, W = determine_input_size(model)

    target_classes = args.classes if args.classes else DEFAULT_SAMPLE_CLASSES
    used_classes, samples = collect_samples(args.test_dir, target_classes, args.per_class)

    if not samples:
        print('No test samples found to evaluate.')
        return

    print(f'Evaluating {len(samples)} samples across {len(used_classes)} classes...')
    for p in samples:
        evaluate_one(model, class_mapping, p, W, H)


if __name__ == '__main__':
    main()