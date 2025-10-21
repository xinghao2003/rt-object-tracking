"""
Data Preparation Utilities for YOLOv11
"""

import os
import shutil
import yaml
from pathlib import Path
import random
import argparse
from tqdm import tqdm


def split_dataset(
    source_images,
    source_labels,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Split dataset into train/val/test sets

    Args:
        source_images: Directory containing images
        source_labels: Directory containing label files
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    source_images = Path(source_images)
    source_labels = Path(source_labels)
    output_dir = Path(output_dir)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"

    # Get list of images
    image_files = sorted([f for f in source_images.glob('*')
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {source_images}")

    print(f"Found {len(image_files)} images")

    # Shuffle images
    random.shuffle(image_files)

    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(files, split):
        for img_file in tqdm(files, desc=f"Copying {split}"):
            # Copy image
            dst_img = output_dir / split / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)

            # Copy corresponding label if exists
            label_file = source_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_dir / split / 'labels' / f"{img_file.stem}.txt"
                shutil.copy2(label_file, dst_label)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset split completed! Output: {output_dir}")


def create_dataset_yaml(
    data_dir,
    class_names,
    output_file=None
):
    """
    Create YOLO dataset YAML file

    Args:
        data_dir: Root directory of dataset
        class_names: List of class names or dict {id: name}
        output_file: Output YAML file path (default: data_dir/dataset.yaml)
    """

    data_dir = Path(data_dir)

    if output_file is None:
        output_file = data_dir / 'dataset.yaml'

    # Handle class names
    if isinstance(class_names, list):
        names_dict = {i: name for i, name in enumerate(class_names)}
    else:
        names_dict = class_names

    # Create YAML config
    config = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(names_dict),  # Number of classes
        'names': names_dict
    }

    # Save YAML
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset YAML created: {output_file}")
    return output_file


def validate_dataset(data_dir):
    """
    Validate dataset structure and report statistics

    Args:
        data_dir: Root directory of dataset
    """

    data_dir = Path(data_dir)

    print("=" * 80)
    print("DATASET VALIDATION REPORT")
    print("=" * 80)

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Set:")
        print("-" * 40)

        img_dir = data_dir / split / 'images'
        lbl_dir = data_dir / split / 'labels'

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} does not exist")
            continue

        images = list(img_dir.glob('*.[jp][pn]g')) + list(img_dir.glob('*.bmp'))
        labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []

        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")

        # Check for missing labels
        missing_labels = []
        for img in images:
            label_file = lbl_dir / f"{img.stem}.txt"
            if not label_file.exists():
                missing_labels.append(img.name)

        if missing_labels:
            print(f"  WARNING: {len(missing_labels)} images without labels")
            if len(missing_labels) <= 5:
                print(f"    Examples: {missing_labels}")

        # Count objects
        total_objects = 0
        class_counts = {}

        if lbl_dir.exists():
            for label_file in labels:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_objects += 1
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print(f"  Total Objects: {total_objects}")
        if class_counts:
            print(f"  Classes: {sorted(class_counts.keys())}")
            for class_id, count in sorted(class_counts.items()):
                print(f"    Class {class_id}: {count} objects")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Dataset Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--images', required=True, help='Source images directory')
    split_parser.add_argument('--labels', required=True, help='Source labels directory')
    split_parser.add_argument('--output', required=True, help='Output directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio')
    split_parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio')
    split_parser.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Create YAML command
    yaml_parser = subparsers.add_parser('create-yaml', help='Create dataset YAML')
    yaml_parser.add_argument('--data-dir', required=True, help='Dataset directory')
    yaml_parser.add_argument('--classes', required=True, nargs='+',
                            help='Class names (e.g., person car dog)')
    yaml_parser.add_argument('--output', help='Output YAML file')

    # Validate dataset command
    val_parser = subparsers.add_parser('validate', help='Validate dataset')
    val_parser.add_argument('--data-dir', required=True, help='Dataset directory')

    args = parser.parse_args()

    if args.command == 'split':
        split_dataset(
            args.images,
            args.labels,
            args.output,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )

    elif args.command == 'create-yaml':
        create_dataset_yaml(
            args.data_dir,
            args.classes,
            args.output
        )

    elif args.command == 'validate':
        validate_dataset(args.data_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
