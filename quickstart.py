#!/usr/bin/env python3
"""
Quick Start Script for YOLOv11 Framework
Helps users get started with training and inference
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_step(step, text):
    """Print step information"""
    print(f"\n[Step {step}] {text}")
    print("-" * 80)


def check_dependencies():
    """Check if dependencies are installed"""
    print_step(1, "Checking Dependencies")

    try:
        import ultralytics
        import torch
        import cv2
        import PIL
        import tkinter
        print("All dependencies are installed!")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - Ultralytics: {ultralytics.__version__}")
        print(f"  - OpenCV: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


def setup_directories():
    """Create necessary directories"""
    print_step(2, "Setting Up Directories")

    dirs = [
        "data/train/images",
        "data/train/labels",
        "data/val/images",
        "data/val/labels",
        "data/test/images",
        "data/test/labels",
        "models",
        "logs",
        "reports",
        "config"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("All directories created successfully!")


def download_sample_model():
    """Download a sample YOLOv11 model"""
    print_step(3, "Downloading Sample Model")

    from ultralytics import YOLO

    print("Downloading YOLOv11n (nano) model...")
    model = YOLO('yolov11n.pt')
    print(f"Model downloaded and ready!")


def show_menu():
    """Show interactive menu"""
    print_header("YOLOv11 Framework - Quick Start")

    print("What would you like to do?")
    print("\n1. Setup environment (check dependencies + create directories)")
    print("2. Download sample YOLOv11 model")
    print("3. Launch GUI application")
    print("4. View example training command")
    print("5. View example validation command")
    print("6. View data preparation commands")
    print("7. Show README")
    print("0. Exit")

    choice = input("\nEnter your choice (0-7): ").strip()
    return choice


def launch_gui():
    """Launch the GUI application"""
    print_step("GUI", "Launching GUI Application")

    gui_script = Path("src/gui/app.py")
    if not gui_script.exists():
        print(f"Error: {gui_script} not found!")
        return

    print("Starting GUI application...")
    print("(Close the GUI window to return to this menu)")
    subprocess.run([sys.executable, str(gui_script)])


def show_training_example():
    """Show example training command"""
    print_step("Example", "Training Command")

    print("Basic training command:")
    print("""
python src/training/train.py \\
    --data ./data/dataset.yaml \\
    --model-size n \\
    --epochs 100 \\
    --batch 16 \\
    --project ./models \\
    --name my_first_model
    """)

    print("\nAdvanced training command:")
    print("""
python src/training/train.py \\
    --data ./data/dataset.yaml \\
    --model-size s \\
    --task detect \\
    --epochs 200 \\
    --batch 32 \\
    --imgsz 640 \\
    --device 0 \\
    --project ./models \\
    --name yolov11s_custom
    """)

    print("\nModel sizes: n (fastest) < s < m < l < x (most accurate)")
    print("Tasks: detect (boxes), segment (masks), pose, classify")


def show_validation_example():
    """Show example validation command"""
    print_step("Example", "Validation Command")

    print("Validate model:")
    print("""
python src/training/validate.py \\
    --model ./models/my_first_model/weights/best.pt \\
    --data ./data/dataset.yaml \\
    --split val
    """)

    print("\nTest model:")
    print("""
python src/training/validate.py \\
    --model ./models/my_first_model/weights/best.pt \\
    --data ./data/dataset.yaml \\
    --split test \\
    --conf 0.25 \\
    --iou 0.45
    """)


def show_data_prep_examples():
    """Show data preparation examples"""
    print_step("Example", "Data Preparation Commands")

    print("Split dataset:")
    print("""
python src/utils/data_utils.py split \\
    --images /path/to/all/images \\
    --labels /path/to/all/labels \\
    --output ./data \\
    --train-ratio 0.7 \\
    --val-ratio 0.2 \\
    --test-ratio 0.1
    """)

    print("\nCreate dataset YAML:")
    print("""
python src/utils/data_utils.py create-yaml \\
    --data-dir ./data \\
    --classes person car bicycle \\
    --output ./data/dataset.yaml
    """)

    print("\nValidate dataset:")
    print("""
python src/utils/data_utils.py validate --data-dir ./data
    """)


def show_readme():
    """Display README content"""
    readme = Path("README.md")
    if readme.exists():
        with open(readme, 'r') as f:
            content = f.read()
        print("\n" + content)
    else:
        print("README.md not found!")


def main():
    """Main function"""

    while True:
        choice = show_menu()

        if choice == '0':
            print("\nGoodbye!")
            break

        elif choice == '1':
            if check_dependencies():
                setup_directories()
                print("\nEnvironment setup complete!")

        elif choice == '2':
            download_sample_model()

        elif choice == '3':
            launch_gui()

        elif choice == '4':
            show_training_example()

        elif choice == '5':
            show_validation_example()

        elif choice == '6':
            show_data_prep_examples()

        elif choice == '7':
            show_readme()

        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()
