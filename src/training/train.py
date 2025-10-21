"""
YOLOv11 Training Script with Comprehensive Logging
Supports training with images and videos
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
import argparse

from ultralytics import YOLO
import torch


class TrainingLogger:
    """Custom logger for training process"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this training session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"train_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        # Setup file logging
        log_file = self.session_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Metrics tracking
        self.metrics = {
            'session_id': self.timestamp,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'config': {}
        }

    def log_config(self, config):
        """Log training configuration"""
        self.metrics['config'] = config
        self.logger.info("Training Configuration:")
        self.logger.info(json.dumps(config, indent=2))

    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        self.metrics['epochs'].append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        self.logger.info(f"Epoch {epoch}: {metrics}")

    def save_metrics(self):
        """Save metrics to JSON file"""
        self.metrics['end_time'] = datetime.now().isoformat()
        metrics_file = self.session_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")


def prepare_dataset_yaml(data_path, task='detect'):
    """
    Prepare dataset YAML configuration for YOLO

    Args:
        data_path: Path to dataset directory
        task: Task type ('detect', 'segment', 'pose', 'classify')

    Returns:
        Path to dataset YAML file
    """
    data_path = Path(data_path)

    # Create dataset YAML
    dataset_config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {}  # Will be populated based on actual classes
    }

    # Try to detect classes from labels
    train_labels = data_path / 'train' / 'labels'
    if train_labels.exists():
        # Get unique classes from label files
        classes = set()
        for label_file in train_labels.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)

        # Create class names (you should customize these)
        dataset_config['names'] = {i: f'class_{i}' for i in sorted(classes)}
    else:
        # Default single class
        dataset_config['names'] = {0: 'object'}

    # Save YAML
    yaml_path = data_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return yaml_path


def train_yolo(
    model_size='n',
    task='detect',
    data_yaml=None,
    epochs=100,
    imgsz=640,
    batch=16,
    device='',
    project='./models',
    name='yolov11_custom',
    pretrained=True,
    resume=False,
    augment=True,
    log_dir='./logs'
):
    """
    Train YOLOv11 model with comprehensive logging

    Args:
        model_size: Model size (n, s, m, l, x)
        task: Task type (detect, segment, pose, classify)
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to use ('' for auto, 'cpu', '0', '0,1', etc.)
        project: Project directory to save results
        name: Run name
        pretrained: Use pretrained weights
        resume: Resume from last checkpoint
        augment: Use data augmentation
        log_dir: Directory for custom logs
    """

    # Initialize logger
    logger = TrainingLogger(log_dir)

    # Prepare configuration
    config = {
        'model_size': model_size,
        'task': task,
        'data_yaml': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device if device else 'auto',
        'project': project,
        'name': name,
        'pretrained': pretrained,
        'resume': resume,
        'augment': augment
    }

    logger.log_config(config)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.logger.info("CUDA not available, using CPU")

    try:
        # Initialize model
        model_name = f'yolov11{model_size}'
        if task == 'segment':
            model_name += '-seg'
        elif task == 'pose':
            model_name += '-pose'
        elif task == 'classify':
            model_name += '-cls'

        model_name += '.pt'

        logger.logger.info(f"Initializing model: {model_name}")
        model = YOLO(model_name if pretrained else None)

        # Start training
        logger.logger.info("Starting training...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=pretrained,
            resume=resume,
            augment=augment,
            verbose=True,
            plots=True,
            save=True,
            save_period=10  # Save checkpoint every 10 epochs
        )

        logger.logger.info("Training completed successfully!")
        logger.logger.info(f"Best model saved to: {model.trainer.best}")

        # Save final metrics
        logger.save_metrics()

        return results

    except Exception as e:
        logger.logger.error(f"Training failed: {str(e)}", exc_info=True)
        logger.save_metrics()
        raise


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 Model')

    # Model parameters
    parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--task', type=str, default='detect',
                        choices=['detect', 'segment', 'pose', 'classify'],
                        help='Task type')

    # Data parameters
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML or directory')
    parser.add_argument('--prepare-yaml', action='store_true',
                        help='Auto-generate dataset YAML from data directory')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='',
                        help='Device (empty=auto, cpu, 0, 0,1, etc.)')

    # Output parameters
    parser.add_argument('--project', type=str, default='./models',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11_custom',
                        help='Run name')

    # Options
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch (no pretrained weights)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Custom log directory')

    args = parser.parse_args()

    # Prepare data YAML if needed
    data_yaml = args.data
    if args.prepare_yaml or not args.data.endswith('.yaml'):
        print(f"Preparing dataset YAML from {args.data}...")
        data_yaml = prepare_dataset_yaml(args.data, args.task)
        print(f"Dataset YAML created at {data_yaml}")

    # Train model
    train_yolo(
        model_size=args.model_size,
        task=args.task,
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        resume=args.resume,
        augment=not args.no_augment,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()
