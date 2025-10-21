# YOLOv11 Object Detection & Tracking Framework

A comprehensive framework for training, validating, and deploying YOLOv11 models with a user-friendly Tkinter GUI application. Supports detection, segmentation, and real-time object tracking with complete logging and reporting capabilities.

## Features

- **Complete Training Pipeline**: Fine-tune YOLOv11 models with comprehensive logging
- **Validation & Testing**: Detailed metrics logging and evaluation
- **GUI Application**: Tkinter-based interface for easy inference
- **Multiple Tasks**: Detection, Segmentation, and Tracking
- **Real-time Visualization**: Live preview with bounding boxes and tracking lines
- **Comprehensive Reporting**: Training analysis and performance metrics
- **Flexible Input**: Support for both images and videos

## Repository Structure

```
rt-object-tracking/
├── config/                      # Configuration files
│   ├── default_config.yaml      # Default training configuration
│   └── example_dataset.yaml     # Example dataset configuration
├── data/                        # Dataset directory
│   ├── train/                   # Training data
│   │   ├── images/
│   │   └── labels/
│   ├── val/                     # Validation data
│   │   ├── images/
│   │   └── labels/
│   └── test/                    # Test data
│       ├── images/
│       └── labels/
├── models/                      # Trained models directory
├── logs/                        # Training/validation logs
├── reports/                     # Analysis reports
├── src/                         # Source code
│   ├── training/                # Training scripts
│   │   ├── train.py            # Main training script
│   │   └── validate.py         # Validation/testing script
│   ├── gui/                     # GUI application
│   │   └── app.py              # Tkinter application
│   └── utils/                   # Utility scripts
│       ├── data_utils.py       # Data preparation utilities
│       └── report_utils.py     # Reporting utilities
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rt-object-tracking
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

#### Option A: Use existing dataset structure
Organize your data in YOLO format:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### Option B: Split existing dataset
```bash
python src/utils/data_utils.py split \
    --images /path/to/all/images \
    --labels /path/to/all/labels \
    --output ./data \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

#### Create dataset YAML configuration:
```bash
python src/utils/data_utils.py create-yaml \
    --data-dir ./data \
    --classes person car bicycle \
    --output ./data/dataset.yaml
```

#### Validate your dataset:
```bash
python src/utils/data_utils.py validate --data-dir ./data
```

### 2. Train a Model

#### Basic training:
```bash
python src/training/train.py \
    --data ./data/dataset.yaml \
    --model-size n \
    --epochs 100 \
    --batch 16 \
    --project ./models \
    --name my_model
```

#### Training with custom parameters:
```bash
python src/training/train.py \
    --data ./data/dataset.yaml \
    --model-size s \
    --task detect \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --project ./models \
    --name yolov11s_custom
```

#### Model sizes:
- `n` (nano): Fastest, smallest
- `s` (small): Balanced speed/accuracy
- `m` (medium): Higher accuracy
- `l` (large): Even better accuracy
- `x` (xlarge): Best accuracy, slowest

#### Task types:
- `detect`: Object detection (bounding boxes)
- `segment`: Instance segmentation (masks)
- `pose`: Pose estimation
- `classify`: Image classification

### 3. Validate/Test Model

#### Validate on validation set:
```bash
python src/training/validate.py \
    --model ./models/my_model/weights/best.pt \
    --data ./data/dataset.yaml \
    --split val
```

#### Test on test set:
```bash
python src/training/validate.py \
    --model ./models/my_model/weights/best.pt \
    --data ./data/dataset.yaml \
    --split test \
    --conf 0.25 \
    --iou 0.45
```

### 4. Run GUI Application

Launch the Tkinter GUI for interactive inference:

```bash
python src/gui/app.py
```

#### GUI Usage:

1. **Load Model**: Click "Browse Model" and select your trained `.pt` file
2. **Select Input**: Choose Image or Video, then browse to your file
3. **Configure Task**: Select Detection, Segmentation, or Tracking
4. **Adjust Parameters**: Set confidence and IOU thresholds
5. **Run Inference**: Click "Run Inference" to start processing

#### GUI Features:

- **Image Mode**:
  - Detection with bounding boxes
  - Segmentation with masks
  - Instant results

- **Video Mode**:
  - Real-time processing
  - Object tracking with ByteTrack
  - Tracking lines visualization
  - Frame-by-frame analysis

### 5. Generate Reports

#### Compare validation sessions:
```bash
python src/utils/report_utils.py compare-validation \
    --log-dir ./logs \
    --report-dir ./reports \
    --plot
```

#### Generate summary report:
```bash
python src/utils/report_utils.py summary \
    --log-dir ./logs \
    --report-dir ./reports
```

## Advanced Usage

### Training with Video Data

YOLOv11 can extract frames from videos for training. Organize videos in your dataset:

```python
# Extract frames from video
import cv2
from pathlib import Path

video_path = "path/to/video.mp4"
output_dir = Path("data/train/images")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save every 10th frame
    if frame_count % 10 == 0:
        cv2.imwrite(str(output_dir / f"frame_{frame_count:06d}.jpg"), frame)

    frame_count += 1

cap.release()
```

Then annotate the extracted frames using tools like:
- [LabelImg](https://github.com/tzutalin/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Roboflow](https://roboflow.com)

### Custom Training Configuration

Edit `config/default_config.yaml` for advanced training options:

```yaml
train:
  epochs: 200
  batch: 32
  imgsz: 640
  optimizer: 'AdamW'
  lr0: 0.001

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
```

### Transfer Learning

Start from a pre-trained model:

```bash
python src/training/train.py \
    --data ./data/dataset.yaml \
    --model-size m \
    --epochs 50 \
    --batch 16
```

The `--no-pretrained` flag trains from scratch (not recommended unless you have a large dataset).

### Resume Training

Resume from a checkpoint:

```bash
python src/training/train.py \
    --data ./data/dataset.yaml \
    --resume \
    --name my_model
```

## Tracking Configuration

The tracking feature uses ByteTrack by default. You can customize tracking parameters:

```yaml
# In your training config or as CLI arguments
tracking:
  tracker: 'bytetrack.yaml'  # or 'botsort.yaml'
  persist: true
  conf: 0.25
```

## Logging and Monitoring

### Training Logs

All training sessions are logged to:
- `logs/train_YYYYMMDD_HHMMSS/training.log` - Detailed logs
- `logs/train_YYYYMMDD_HHMMSS/metrics.json` - Metrics data
- `models/your_model/` - Ultralytics default outputs
  - `weights/best.pt` - Best model checkpoint
  - `weights/last.pt` - Last epoch checkpoint
  - `results.png` - Training curves
  - `confusion_matrix.png` - Confusion matrix

### Validation Logs

Validation results are logged to:
- `logs/val_YYYYMMDD_HHMMSS/validation.log`
- `logs/val_YYYYMMDD_HHMMSS/metrics.json`
- `reports/validation_report_YYYYMMDD_HHMMSS.txt`

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir ./models/your_model
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python src/training/train.py --batch 8 ...
```

Or reduce image size:
```bash
python src/training/train.py --imgsz 416 ...
```

### GUI Not Showing Video

Ensure OpenCV is properly installed:
```bash
pip install opencv-python opencv-contrib-python --upgrade
```

### Low Accuracy

Try:
1. Increase training epochs
2. Use larger model size (s, m, l instead of n)
3. Increase dataset size
4. Adjust data augmentation
5. Check dataset quality and labels

## Performance Tips

1. **Use GPU**: Training on GPU is 10-100x faster
2. **Batch Size**: Larger batches = faster training (if GPU memory allows)
3. **Image Size**: 640 is standard, but 416 is faster with slightly lower accuracy
4. **Model Size**: Start with 'n' or 's' for quick experiments, use 'l' or 'x' for production

## Example Workflow

Complete workflow from dataset to deployment:

```bash
# 1. Prepare dataset
python src/utils/data_utils.py split \
    --images ./raw_data/images \
    --labels ./raw_data/labels \
    --output ./data

# 2. Create YAML
python src/utils/data_utils.py create-yaml \
    --data-dir ./data \
    --classes person car truck

# 3. Validate dataset
python src/utils/data_utils.py validate --data-dir ./data

# 4. Train model
python src/training/train.py \
    --data ./data/dataset.yaml \
    --model-size s \
    --epochs 100 \
    --batch 16 \
    --name vehicle_detector

# 5. Validate model
python src/training/validate.py \
    --model ./models/vehicle_detector/weights/best.pt \
    --data ./data/dataset.yaml \
    --split val

# 6. Test model
python src/training/validate.py \
    --model ./models/vehicle_detector/weights/best.pt \
    --data ./data/dataset.yaml \
    --split test

# 7. Generate reports
python src/utils/report_utils.py summary

# 8. Run GUI
python src/gui/app.py
```

## Citation

If you use this framework, please cite:

```bibtex
@software{yolov11_framework,
  title = {YOLOv11 Object Detection & Tracking Framework},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/rt-object-tracking}
}
```

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the documentation above
2. Review [Ultralytics documentation](https://docs.ultralytics.com/)
3. Open an issue on GitHub

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Roadmap

- [ ] Export to ONNX/TensorRT
- [ ] Multi-camera support
- [ ] REST API for inference
- [ ] Docker deployment
- [ ] Cloud training support
- [ ] Auto-labeling tools
