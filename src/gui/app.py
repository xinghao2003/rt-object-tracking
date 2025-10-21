"""
YOLOv11 Tkinter GUI Application
Features: Detection, Segmentation, Tracking with real-time preview
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
import time

from ultralytics import YOLO


class YOLOApp:
    """Main application class for YOLOv11 GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 Object Detection & Tracking")
        self.root.geometry("1400x800")

        # Application state
        self.model = None
        self.current_model_path = None
        self.current_source = None
        self.source_type = None  # 'image' or 'video'
        self.is_processing = False
        self.cap = None
        self.track_history = {}  # Store tracking history for visualization

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the user interface"""

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Configuration
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Preview
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create configuration panel
        self.create_config_panel(left_panel)

        # Create preview panel
        self.create_preview_panel(right_panel)

    def create_config_panel(self, parent):
        """Create configuration side panel"""

        # Title
        title = ttk.Label(parent, text="Configuration Panel",
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 20))

        # Model Selection Section
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var,
                               state='readonly')
        model_entry.pack(fill=tk.X, pady=(0, 5))

        model_btn = ttk.Button(model_frame, text="Browse Model",
                              command=self.browse_model)
        model_btn.pack(fill=tk.X)

        # Input Source Section
        source_frame = ttk.LabelFrame(parent, text="Input Source", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 10))

        self.source_type_var = tk.StringVar(value='image')
        ttk.Radiobutton(source_frame, text="Image", variable=self.source_type_var,
                       value='image', command=self.on_source_type_change).pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Video", variable=self.source_type_var,
                       value='video', command=self.on_source_type_change).pack(anchor=tk.W)

        ttk.Separator(source_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        self.source_path_var = tk.StringVar()
        source_entry = ttk.Entry(source_frame, textvariable=self.source_path_var,
                                state='readonly')
        source_entry.pack(fill=tk.X, pady=(0, 5))

        source_btn = ttk.Button(source_frame, text="Browse Source",
                               command=self.browse_source)
        source_btn.pack(fill=tk.X)

        # Task Selection Section
        task_frame = ttk.LabelFrame(parent, text="Task Type", padding=10)
        task_frame.pack(fill=tk.X, pady=(0, 10))

        self.task_var = tk.StringVar(value='detect')
        ttk.Radiobutton(task_frame, text="Detection", variable=self.task_var,
                       value='detect').pack(anchor=tk.W)
        ttk.Radiobutton(task_frame, text="Segmentation", variable=self.task_var,
                       value='segment').pack(anchor=tk.W)
        ttk.Radiobutton(task_frame, text="Tracking (Video only)",
                       variable=self.task_var, value='track').pack(anchor=tk.W)

        # Inference Parameters Section
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Confidence threshold
        ttk.Label(params_frame, text="Confidence:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_scale = ttk.Scale(params_frame, from_=0.0, to=1.0,
                              variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X)
        self.conf_label = ttk.Label(params_frame, text="0.25")
        self.conf_label.pack(anchor=tk.W)
        self.conf_var.trace('w', self.update_conf_label)

        # IOU threshold
        ttk.Label(params_frame, text="IOU Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(params_frame, from_=0.0, to=1.0,
                             variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.pack(fill=tk.X)
        self.iou_label = ttk.Label(params_frame, text="0.45")
        self.iou_label.pack(anchor=tk.W)
        self.iou_var.trace('w', self.update_iou_label)

        # Show tracking lines (for video tracking)
        self.show_tracks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Show Tracking Lines",
                       variable=self.show_tracks_var).pack(anchor=tk.W, pady=(10, 0))

        # Control Buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(20, 0))

        self.run_btn = ttk.Button(control_frame, text="Run Inference",
                                  command=self.run_inference, state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_btn = ttk.Button(control_frame, text="Stop",
                                   command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)

        # Status Section
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD,
                                   state=tk.DISABLED, font=('Courier', 9))
        status_scroll = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_preview_panel(self, parent):
        """Create preview panel for displaying results"""

        # Title
        title = ttk.Label(parent, text="Preview Window",
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 10))

        # Canvas for image/video display
        self.canvas = tk.Canvas(parent, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text
        self.placeholder_text = self.canvas.create_text(
            400, 300, text="No image loaded",
            fill='white', font=('Arial', 16)
        )

    def update_conf_label(self, *args):
        """Update confidence label"""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    def update_iou_label(self, *args):
        """Update IOU label"""
        self.iou_label.config(text=f"{self.iou_var.get():.2f}")

    def on_source_type_change(self):
        """Handle source type change"""
        self.source_path_var.set("")
        self.current_source = None

    def browse_model(self):
        """Browse and load model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                self.log_status(f"Loading model: {file_path}")
                self.model = YOLO(file_path)
                self.current_model_path = file_path
                self.model_path_var.set(Path(file_path).name)
                self.log_status("Model loaded successfully!")
                self.update_run_button_state()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.log_status(f"Error loading model: {str(e)}")

    def browse_source(self):
        """Browse and load source (image or video)"""
        source_type = self.source_type_var.get()

        if source_type == 'image':
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                          ("All Files", "*.*")]
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                          ("All Files", "*.*")]
            )

        if file_path:
            self.current_source = file_path
            self.source_type = source_type
            self.source_path_var.set(Path(file_path).name)
            self.log_status(f"Source loaded: {file_path}")
            self.update_run_button_state()

            # Preview the source
            if source_type == 'image':
                self.preview_image(file_path)

    def update_run_button_state(self):
        """Enable/disable run button based on state"""
        if self.model and self.current_source:
            self.run_btn.config(state=tk.NORMAL)
        else:
            self.run_btn.config(state=tk.DISABLED)

    def log_status(self, message):
        """Log message to status text widget"""
        self.status_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def preview_image(self, image_path):
        """Preview image on canvas"""
        try:
            # Load and resize image
            img = Image.open(image_path)
            img = self.resize_image_to_canvas(img)

            # Display on canvas
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.photo, anchor=tk.CENTER
            )
        except Exception as e:
            self.log_status(f"Error previewing image: {str(e)}")

    def resize_image_to_canvas(self, img):
        """Resize image to fit canvas while maintaining aspect ratio"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 600

        img_width, img_height = img.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)

        new_width = int(img_width * ratio * 0.95)
        new_height = int(img_height * ratio * 0.95)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def run_inference(self):
        """Run inference on selected source"""
        if not self.model or not self.current_source:
            return

        task = self.task_var.get()

        # Validate task
        if task == 'track' and self.source_type != 'video':
            messagebox.showwarning("Warning",
                                  "Tracking is only available for video sources!")
            return

        # Start processing in separate thread
        self.is_processing = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        thread = threading.Thread(target=self.process_source, daemon=True)
        thread.start()

    def process_source(self):
        """Process source (image or video) in separate thread"""
        try:
            if self.source_type == 'image':
                self.process_image()
            else:
                self.process_video()
        except Exception as e:
            self.log_status(f"Processing error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
        finally:
            self.is_processing = False
            self.run_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def process_image(self):
        """Process single image"""
        task = self.task_var.get()
        conf = self.conf_var.get()
        iou = self.iou_var.get()

        self.log_status(f"Processing image with {task}...")

        # Run inference
        if task == 'segment':
            results = self.model.predict(
                self.current_source,
                conf=conf,
                iou=iou,
                show_labels=True,
                show_conf=True
            )
        else:
            results = self.model.predict(
                self.current_source,
                conf=conf,
                iou=iou,
                show_labels=True,
                show_conf=True
            )

        # Get annotated image
        if results and len(results) > 0:
            annotated = results[0].plot()

            # Convert BGR to RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Display result
            self.display_frame(annotated_rgb)

            # Log detections
            num_detections = len(results[0].boxes)
            self.log_status(f"Detected {num_detections} objects")

    def process_video(self):
        """Process video with optional tracking"""
        task = self.task_var.get()
        conf = self.conf_var.get()
        iou = self.iou_var.get()

        self.log_status(f"Processing video with {task}...")

        # Open video
        self.cap = cv2.VideoCapture(self.current_source)

        if not self.cap.isOpened():
            self.log_status("Error: Could not open video")
            return

        frame_count = 0
        self.track_history = {}

        while self.is_processing and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                self.log_status("Video processing complete")
                break

            frame_count += 1

            # Run inference
            if task == 'track':
                results = self.model.track(
                    frame,
                    conf=conf,
                    iou=iou,
                    persist=True,
                    tracker="bytetrack.yaml"
                )
            else:
                results = self.model.predict(
                    frame,
                    conf=conf,
                    iou=iou
                )

            # Annotate frame
            if results and len(results) > 0:
                annotated = results[0].plot()

                # Draw tracking lines if enabled
                if task == 'track' and self.show_tracks_var.get():
                    annotated = self.draw_tracking_lines(annotated, results[0])

                # Convert BGR to RGB
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Display frame
                self.display_frame(annotated_rgb)

            # Control frame rate
            time.sleep(0.03)  # ~30 fps

        # Release video capture
        if self.cap:
            self.cap.release()
            self.cap = None

    def draw_tracking_lines(self, frame, result):
        """Draw tracking lines on frame"""
        if not hasattr(result, 'boxes') or result.boxes.id is None:
            return frame

        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Update tracking history
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center = (int(x), int(y))

            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append(center)

            # Keep only recent history (last 30 points)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Draw tracking line
            points = np.array(self.track_history[track_id], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(frame, [points], False, (0, 255, 0), 2)

        return frame

    def display_frame(self, frame_rgb):
        """Display frame on canvas"""
        try:
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            img = self.resize_image_to_canvas(img)

            # Display on canvas
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.photo, anchor=tk.CENTER
            )

            # Update UI
            self.root.update()

        except Exception as e:
            self.log_status(f"Display error: {str(e)}")

    def stop_processing(self):
        """Stop current processing"""
        self.is_processing = False
        self.log_status("Stopping processing...")

        if self.cap:
            self.cap.release()
            self.cap = None

    def on_closing(self):
        """Handle window closing"""
        self.stop_processing()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
