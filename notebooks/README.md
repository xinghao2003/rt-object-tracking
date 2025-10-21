# Training Notebooks

This directory contains Jupyter notebooks for training YOLOv11 models, optimized for cloud platforms like Google Colab.

## Available Notebooks

### 1. train_yolov11_colab.ipynb

Complete training notebook optimized for Google Colab with free GPU access.

**Features:**
- GPU setup and verification
- Multiple dataset upload methods (ZIP, Google Drive, URL)
- Automatic dataset splitting and validation
- Comprehensive training pipeline
- Real-time monitoring with TensorBoard
- Validation and testing
- Model export (ONNX, TensorRT, etc.)
- Download trained models

**Quick Start:**
1. Click the "Open in Colab" badge in the main README
2. Or upload the notebook to [Google Colab](https://colab.research.google.com/)
3. Enable GPU: Runtime → Change runtime type → GPU (T4, A100, or V100)
4. Follow the step-by-step instructions

**Typical Usage:**

```python
# 1. Setup (run these cells first)
- Check GPU availability
- Install dependencies
- Clone repository or mount Google Drive

# 2. Prepare Dataset
- Upload ZIP file
- Or use dataset from Google Drive
- Or download from URL (Roboflow, Kaggle)
- Split dataset if needed
- Create dataset YAML configuration
- Validate dataset structure

# 3. Configure Training
CONFIG = {
    'model_size': 'n',      # n, s, m, l, x
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'data': 'data/dataset.yaml'
}

# 4. Train Model
- Initialize model
- Start training
- Monitor with TensorBoard

# 5. Validate & Test
- Run validation on val set
- Run test on test set
- View metrics and plots

# 6. Export & Download
- Export to ONNX (optional)
- Download trained model
- Or save to Google Drive
```

## Google Colab Tips

### GPU Selection
- Free tier: Tesla T4 (16GB VRAM)
- Colab Pro: T4, V100, A100
- Check availability: `!nvidia-smi`

### Runtime Limits
- Free tier: ~12 hours per session
- Save checkpoints regularly
- Use Google Drive for persistence

### Memory Management
If you get OOM (Out of Memory) errors:
```python
# Reduce batch size
CONFIG['batch'] = 8  # or even 4

# Use smaller image size
CONFIG['imgsz'] = 416  # instead of 640

# Use smaller model
CONFIG['model_size'] = 'n'  # nano is smallest
```

### Speed Optimization
```python
# Enable mixed precision training (automatic in YOLOv11)
# Cache images for faster loading
CONFIG['cache'] = True

# Increase workers if CPU allows
CONFIG['workers'] = 8
```

## Dataset Upload Methods

### Method 1: ZIP File
1. Prepare your dataset in YOLO format
2. Create ZIP file: `dataset.zip`
3. Upload in Colab: Use the upload cell
4. Notebook will extract automatically

### Method 2: Google Drive
1. Upload dataset to Google Drive
2. Mount Drive in Colab: `drive.mount('/content/drive')`
3. Point to your dataset path

### Method 3: Download from URL
```python
# From direct URL
!wget https://url-to-dataset.zip -O dataset.zip

# From Roboflow
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("project-name")
dataset = project.version(1).download("yolov8")

# From Kaggle
!pip install kaggle
!kaggle datasets download -d dataset-name
```

## Expected Dataset Structure

```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Training Examples

### Quick Training (Fast Experiment)
```python
CONFIG = {
    'model_size': 'n',
    'epochs': 50,
    'batch': 16,
    'imgsz': 640
}
```
**Time:** ~15-30 minutes on T4

### Standard Training
```python
CONFIG = {
    'model_size': 's',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640
}
```
**Time:** ~60-120 minutes on T4

### High Accuracy Training
```python
CONFIG = {
    'model_size': 'm',
    'epochs': 200,
    'batch': 16,
    'imgsz': 640
}
```
**Time:** ~3-4 hours on T4

### Segmentation Task
```python
CONFIG = {
    'model_size': 's',
    'task': 'segment',  # Instance segmentation
    'epochs': 100,
    'batch': 16,
    'imgsz': 640
}
```

## After Training

Once training is complete:

### 1. Download Model
```python
# Download best model
files.download('models/yolov11_custom/weights/best.pt')

# Or save to Google Drive
!cp models/yolov11_custom/weights/best.pt /content/drive/MyDrive/
```

### 2. Use Locally
```bash
# With GUI
python src/gui/app.py

# With CLI
python src/inference/predict.py --model best.pt --source image.jpg
```

### 3. Deploy
```python
# Export to ONNX for deployment
model.export(format='onnx')

# Export to TensorFlow Lite (mobile)
model.export(format='tflite')

# Export to CoreML (iOS)
model.export(format='coreml')
```

## Troubleshooting

### GPU Not Available
```python
# Check GPU
!nvidia-smi

# If no GPU, go to:
# Runtime → Change runtime type → Hardware accelerator → GPU
```

### Out of Memory
```python
# Reduce batch size
CONFIG['batch'] = 8

# Or reduce image size
CONFIG['imgsz'] = 416
```

### Session Timeout
```python
# Save checkpoints to Google Drive
drive.mount('/content/drive')

# Training will auto-save every N epochs
CONFIG['save_period'] = 10
```

### Slow Training
```python
# Enable caching
CONFIG['cache'] = True

# Use smaller model for testing
CONFIG['model_size'] = 'n'
```

## Additional Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

## Support

For issues with the notebook:
1. Check the troubleshooting section above
2. Review the main repository README
3. Open an issue on GitHub

---

**Happy Training on Colab! 🚀**
