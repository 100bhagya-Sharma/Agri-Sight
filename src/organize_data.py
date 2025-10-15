import os
import shutil
import random
from pathlib import Path
import argparse


def organize_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize data into train/val/test splits while maintaining class structure.
    
    Args:
        source_dir: Source directory containing class folders
        dest_dir: Destination directory for organized data
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
    """
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in class_dirs:
        print(f"Processing class: {class_name}")
        
        # Create class directories in each split
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        
        for dir_path in [class_train_dir, class_val_dir, class_test_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Get all images for this class
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        if n_images == 0:
            print(f"  WARNING: No images for class {class_name}")
            continue
        n_train = int(train_ratio * n_images)
        n_val = int(val_ratio * n_images)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for img_list, target_dir in [
            (train_images, class_train_dir),
            (val_images, class_val_dir),
            (test_images, class_test_dir)
        ]:
            for image in img_list:
                src = os.path.join(class_dir, image)
                dst = os.path.join(target_dir, image)
                shutil.copy2(src, dst)
        
        print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")


def create_subset(source_root, dest_root, classes, per_class=50):
    """Create a subset dataset from an existing split source_root (train/val/test).
    Copies up to per_class images per class for each split into dest_root.
    """
    splits = ['train', 'val', 'test']
    print(f"Creating subset dataset from {source_root} -> {dest_root}")
    print(f"Classes: {', '.join(classes)} | per_class={per_class}")
    
    for split in splits:
        split_src = os.path.join(source_root, split)
        split_dst = os.path.join(dest_root, split)
        os.makedirs(split_dst, exist_ok=True)
        for cls in classes:
            cls_src = os.path.join(split_src, cls)
            cls_dst = os.path.join(split_dst, cls)
            if not os.path.isdir(cls_src):
                print(f"  [SKIP] {split}/{cls} not found")
                continue
            os.makedirs(cls_dst, exist_ok=True)
            files = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(files)
            take = files[:per_class]
            for fname in take:
                shutil.copy2(os.path.join(cls_src, fname), os.path.join(cls_dst, fname))
            print(f"  [{split}] {cls}: copied {len(take)} files")


def main():
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_source_dir = os.path.join(base_dir, 'data', 'processed')
    default_dest_dir = os.path.join(base_dir, 'processed_data')
    default_subset_source = os.path.join(base_dir, 'processed_data')
    default_subset_dest = os.path.join(base_dir, 'processed_data_subset')

    parser = argparse.ArgumentParser(description='Organize dataset and optionally create a subset dataset')
    parser.add_argument('--source_dir', type=str, default=default_source_dir,
                        help='Source directory containing class folders (for organizing raw processed data)')
    parser.add_argument('--dest_dir', type=str, default=default_dest_dir,
                        help='Destination directory for organized train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    # Subset options
    parser.add_argument('--subset_source_dir', type=str, default=default_subset_source,
                        help='Source root with existing train/val/test splits to sample from')
    parser.add_argument('--subset_dest_dir', type=str, default=default_subset_dest,
                        help='Destination root to create subset train/val/test')
    parser.add_argument('--subset_classes', type=str, nargs='*', default=None,
                        help='List of class folder names to include in the subset')
    parser.add_argument('--subset_per_class', type=int, default=50,
                        help='Max number of images per class per split to copy into subset')
    args = parser.parse_args()

    print("Starting data organization...")
    organize_data(args.source_dir, args.dest_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    print("\nData organization completed!")

    # Create subset if classes provided
    if args.subset_classes:
        print("\nStarting subset creation...")
        create_subset(args.subset_source_dir, args.subset_dest_dir, args.subset_classes, args.subset_per_class)
        print("\nSubset creation completed!")


if __name__ == "__main__":
    main()