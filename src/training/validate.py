"""
YOLOv11 Validation and Testing Script
Comprehensive metrics logging and evaluation
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ValidationLogger:
    """Custom logger for validation process"""

    def __init__(self, log_dir, report_dir):
        self.log_dir = Path(log_dir)
        self.report_dir = Path(report_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this validation session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"val_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        # Setup file logging
        log_file = self.session_dir / "validation.log"
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
            'results': {},
            'config': {}
        }

    def log_config(self, config):
        """Log validation configuration"""
        self.metrics['config'] = config
        self.logger.info("Validation Configuration:")
        self.logger.info(json.dumps(config, indent=2))

    def log_results(self, results):
        """Log validation results"""
        self.metrics['results'] = results
        self.logger.info("Validation Results:")
        self.logger.info(json.dumps(results, indent=2))

    def save_metrics(self):
        """Save metrics to JSON file"""
        self.metrics['end_time'] = datetime.now().isoformat()
        metrics_file = self.session_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")

        # Also save as CSV for easy analysis
        self.save_metrics_csv()

    def save_metrics_csv(self):
        """Save metrics as CSV for easy analysis"""
        if 'results' in self.metrics and self.metrics['results']:
            df = pd.DataFrame([self.metrics['results']])
            csv_file = self.session_dir / "metrics.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Metrics CSV saved to {csv_file}")

    def generate_report(self, model_path, results):
        """Generate comprehensive validation report"""
        report_file = self.report_dir / f"validation_report_{self.timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLOv11 VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Model: {model_path}\n\n")

            f.write("-" * 80 + "\n")
            f.write("METRICS SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            for key, value in results.items():
                f.write(f"{key:30s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")

        self.logger.info(f"Report generated: {report_file}")
        return report_file


def validate_model(
    model_path,
    data_yaml,
    split='val',
    imgsz=640,
    batch=16,
    device='',
    save_json=True,
    save_hybrid=True,
    conf_threshold=0.001,
    iou_threshold=0.6,
    log_dir='./logs',
    report_dir='./reports'
):
    """
    Validate YOLOv11 model with comprehensive metrics logging

    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        split: Dataset split to validate on ('val' or 'test')
        imgsz: Input image size
        batch: Batch size
        device: Device to use
        save_json: Save results in JSON format
        save_hybrid: Save hybrid version of labels
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        log_dir: Directory for logs
        report_dir: Directory for reports
    """

    # Initialize logger
    logger = ValidationLogger(log_dir, report_dir)

    # Prepare configuration
    config = {
        'model_path': str(model_path),
        'data_yaml': str(data_yaml),
        'split': split,
        'imgsz': imgsz,
        'batch': batch,
        'device': device if device else 'auto',
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }

    logger.log_config(config)

    try:
        # Load model
        logger.logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)

        # Run validation
        logger.logger.info(f"Running validation on {split} split...")
        results = model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            save_json=save_json,
            save_hybrid=save_hybrid,
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            verbose=True
        )

        # Extract metrics
        metrics_dict = {
            'precision': float(results.box.p.mean()) if hasattr(results.box, 'p') else 0.0,
            'recall': float(results.box.r.mean()) if hasattr(results.box, 'r') else 0.0,
            'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        }

        # Add per-class metrics if available
        if hasattr(results.box, 'maps'):
            for i, map_val in enumerate(results.box.maps):
                metrics_dict[f'mAP50-95_class_{i}'] = float(map_val)

        logger.log_results(metrics_dict)

        # Save metrics
        logger.save_metrics()

        # Generate report
        logger.generate_report(model_path, metrics_dict)

        logger.logger.info("Validation completed successfully!")

        return results, metrics_dict

    except Exception as e:
        logger.logger.error(f"Validation failed: {str(e)}", exc_info=True)
        logger.save_metrics()
        raise


def test_model(
    model_path,
    data_yaml,
    imgsz=640,
    batch=16,
    device='',
    conf_threshold=0.25,
    iou_threshold=0.45,
    log_dir='./logs',
    report_dir='./reports'
):
    """
    Test YOLOv11 model on test set

    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        imgsz: Input image size
        batch: Batch size
        device: Device to use
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        log_dir: Directory for logs
        report_dir: Directory for reports
    """
    return validate_model(
        model_path=model_path,
        data_yaml=data_yaml,
        split='test',
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        log_dir=log_dir,
        report_dir=report_dir
    )


def main():
    parser = argparse.ArgumentParser(description='Validate/Test YOLOv11 Model')

    # Model and data
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')

    # Validation parameters
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='',
                        help='Device (empty=auto, cpu, 0, 0,1, etc.)')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold for NMS')

    # Output
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--report-dir', type=str, default='./reports',
                        help='Report directory')
    parser.add_argument('--no-save-json', action='store_true',
                        help='Do not save results in JSON format')

    args = parser.parse_args()

    # Run validation/testing
    if args.split == 'test':
        test_model(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            log_dir=args.log_dir,
            report_dir=args.report_dir
        )
    else:
        validate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            save_json=not args.no_save_json,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            log_dir=args.log_dir,
            report_dir=args.report_dir
        )


if __name__ == '__main__':
    main()
