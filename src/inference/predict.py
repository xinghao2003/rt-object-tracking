"""
Command-line inference script for YOLOv11
Simple alternative to GUI for quick predictions
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def predict_image(model_path, image_path, conf=0.25, iou=0.45, save=True, save_dir='./results'):
    """
    Run prediction on image

    Args:
        model_path: Path to model file
        image_path: Path to image
        conf: Confidence threshold
        iou: IOU threshold
        save: Save annotated image
        save_dir: Directory to save results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running prediction on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        save=save,
        project=save_dir,
        name='predictions',
        show_labels=True,
        show_conf=True
    )

    if results:
        print(f"\nDetected {len(results[0].boxes)} objects")
        if save:
            print(f"Results saved to: {save_dir}/predictions")

    return results


def predict_video(model_path, video_path, conf=0.25, iou=0.45, save=True, save_dir='./results'):
    """
    Run prediction on video

    Args:
        model_path: Path to model file
        video_path: Path to video
        conf: Confidence threshold
        iou: IOU threshold
        save: Save annotated video
        save_dir: Directory to save results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running prediction on: {video_path}")
    results = model.predict(
        source=video_path,
        conf=conf,
        iou=iou,
        save=save,
        project=save_dir,
        name='predictions',
        show_labels=True,
        show_conf=True,
        stream=True
    )

    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    print(f"\nProcessed {frame_count} frames")
    if save:
        print(f"Results saved to: {save_dir}/predictions")

    return results


def track_video(model_path, video_path, conf=0.25, iou=0.45, tracker='bytetrack.yaml',
                save=True, save_dir='./results'):
    """
    Run tracking on video

    Args:
        model_path: Path to model file
        video_path: Path to video
        conf: Confidence threshold
        iou: IOU threshold
        tracker: Tracker configuration
        save: Save annotated video
        save_dir: Directory to save results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running tracking on: {video_path}")
    results = model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        tracker=tracker,
        save=save,
        project=save_dir,
        name='tracking',
        show_labels=True,
        show_conf=True,
        persist=True,
        stream=True
    )

    frame_count = 0
    total_tracks = set()

    for r in results:
        frame_count += 1

        # Track unique object IDs
        if hasattr(r.boxes, 'id') and r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            total_tracks.update(track_ids)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, {len(total_tracks)} unique objects tracked")

    print(f"\nProcessed {frame_count} frames")
    print(f"Tracked {len(total_tracks)} unique objects")
    if save:
        print(f"Results saved to: {save_dir}/tracking")

    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Inference Script')

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or video file')

    # Task
    parser.add_argument('--task', type=str, default='detect',
                       choices=['detect', 'track'],
                       help='Task type (detect or track)')

    # Parameters
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold')

    # Tracking
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                       help='Tracker configuration (for tracking task)')

    # Output
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')
    parser.add_argument('--no-save', dest='save', action='store_false',
                       help='Do not save results')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Determine source type
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {args.source}")
        return

    # Check if video or image
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    is_video = source_path.suffix.lower() in video_extensions
    is_image = source_path.suffix.lower() in image_extensions

    if not (is_video or is_image):
        print(f"Error: Unsupported file format: {source_path.suffix}")
        return

    # Run appropriate task
    if is_image:
        if args.task == 'track':
            print("Warning: Tracking is only available for videos. Using detection instead.")

        predict_image(
            model_path=args.model,
            image_path=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_dir=args.save_dir
        )

    else:  # video
        if args.task == 'track':
            track_video(
                model_path=args.model,
                video_path=args.source,
                conf=args.conf,
                iou=args.iou,
                tracker=args.tracker,
                save=args.save,
                save_dir=args.save_dir
            )
        else:
            predict_video(
                model_path=args.model,
                video_path=args.source,
                conf=args.conf,
                iou=args.iou,
                save=args.save,
                save_dir=args.save_dir
            )


if __name__ == '__main__':
    main()
