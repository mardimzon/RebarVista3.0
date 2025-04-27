#!/usr/bin/env python3
# Rebar Analysis Application using PyQt5 and external A4Tech camera

import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
import threading
import json
import math
import csv
import traceback
import gc  # Garbage collector
import base64
import io
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Import PyQt5 components
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QFrame, QSplitter, QTextEdit, 
    QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Import Detectron2 libraries
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Global state for API
latest_analysis = {
    "timestamp": None,
    "image": None,
    "image_path": None,
    "segments": [],
    "total_volume": 0
}

# Initialize Flask API
api_app = Flask(__name__)
CORS(api_app)

# Global reference to app instance
app_instance = None
camera_lock = threading.Lock()  # Add a lock for thread-safe camera access

# Worker thread for camera operations
class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        
    def start_camera(self):
        try:
            with camera_lock:
                # Try to access the external webcam (usually at index 0)
                self.camera = cv2.VideoCapture(0)
                
                # Set camera resolution to match original settings (300x400)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
                
                # Check if camera opened successfully
                if not self.camera.isOpened():
                    self.error.emit("Failed to open the camera")
                    return False
                
                print("External A4Tech camera initialized with dimensions (300x400)")
                return True
        except Exception as e:
            self.error.emit(f"Failed to initialize camera: {str(e)}")
            return False
            
    def capture_continuously(self):
        self.running = True
        while self.running:
            try:
                with camera_lock:
                    if self.camera is None or not self.camera.isOpened():
                        time.sleep(0.5)
                        continue
                        
                    ret, frame = self.camera.read()
                
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    self.error.emit("Failed to capture frame")
                    # Try to restart camera
                    self.restart_camera()
            except Exception as e:
                self.error.emit(f"Camera error: {str(e)}")
                time.sleep(1)
            
            # Control frame rate
            time.sleep(0.03)  # ~30fps
            
    def stop(self):
        self.running = False
        with camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                
    def restart_camera(self):
        try:
            with camera_lock:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                
                time.sleep(1)  # Wait before restarting
                
                # Try to access the external webcam again
                self.camera = cv2.VideoCapture(0)
                
                # Set camera resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
                
                if not self.camera.isOpened():
                    self.error.emit("Failed to restart camera")
        except Exception as e:
            self.error.emit(f"Error restarting camera: {str(e)}")
    
    def capture_single_frame(self):
        try:
            # Capture multiple frames to ensure we get a good one
            with camera_lock:
                if self.camera is None or not self.camera.isOpened():
                    self.error.emit("Camera not available")
                    return None
                    
                # Try multiple frames to get a good one
                for _ in range(5):  # Try up to 5 frames
                    ret, frame = self.camera.read()
                    if ret and frame is not None and not np.all(frame == 0):
                        return frame
                    time.sleep(0.1)
                
                self.error.emit("Failed to capture a valid frame")
                return None
        except Exception as e:
            self.error.emit(f"Error capturing frame: {str(e)}")
            return None

# Worker thread for initialization and model loading
class InitWorker(QObject):
    models_loaded = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def load_models(self):
        try:
            self.progress.emit("Loading rebar detection model...")
            
            # Set device explicitly to CPU
            device = "cpu"
            
            # Rebar detection model
            rebar_cfg = get_cfg()
            rebar_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
            rebar_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            rebar_cfg.MODEL.DEVICE = "cpu"
            rebar_model = build_model(rebar_cfg)
            DetectionCheckpointer(rebar_model).load("rebar_model1.pth")
            rebar_model.eval()
            
            self.progress.emit("Loading section detection model...")
            
            # Section detection model
            section_cfg = get_cfg()
            section_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
            section_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            section_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            section_cfg.MODEL.DEVICE = "cpu"
            section_model = build_model(section_cfg)
            DetectionCheckpointer(section_model).load("section_model1.pth") 
            section_model.eval()
            
            # Load cement mixture ratios
            cement_ratios = self.load_cement_ratios()
            
            # Send models back
            models_data = {
                "rebar_model": rebar_model,
                "section_model": section_model,
                "rebar_cfg": rebar_cfg,
                "section_cfg": section_cfg,
                "cement_ratios": cement_ratios
            }
            
            self.models_loaded.emit(models_data)
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print(traceback.format_exc())
            self.error.emit(f"Failed to load models: {str(e)}")
    
    def load_cement_ratios(self):
        """Load cement mixture ratios based on rebar diameter"""
        # Default ratios if file doesn't exist
        cement_ratios = {
            "small": {"cement": 1, "sand": 2, "aggregate": 3, "diameter_range": [6, 12]},
            "medium": {"cement": 1, "sand": 2, "aggregate": 4, "diameter_range": [12, 20]},
            "large": {"cement": 1, "sand": 3, "aggregate": 5, "diameter_range": [20, 50]}
        }
        
        # Try to load from file
        try:
            if os.path.exists('cement_ratios.json'):
                with open('cement_ratios.json', 'r') as f:
                    cement_ratios = json.load(f)
                print("Loaded cement ratios from file")
            else:
                # Create default cement ratios file
                with open('cement_ratios.json', 'w') as f:
                    json.dump(cement_ratios, f, indent=2)
                print("Created default cement ratios file")
        except Exception as e:
            print(f"Error handling cement ratios: {e}")
            
        return cement_ratios
    
    # Worker thread for analysis operations
class AnalysisWorker(QObject):
    analysis_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, rebar_model, section_model, rebar_cfg, section_cfg, cement_ratios):
        super().__init__()
        self.rebar_model = rebar_model
        self.section_model = section_model
        self.rebar_cfg = rebar_cfg
        self.section_cfg = section_cfg
        self.cement_ratios = cement_ratios
        self.captured_frame = None
        self.result_image = None
        self.current_results = []
        self.current_timestamp = None
        self.current_result_dir = None
        
    def set_captured_frame(self, frame, timestamp, result_dir):
        self.captured_frame = frame
        self.current_timestamp = timestamp
        self.current_result_dir = result_dir
        self.current_results = []  # Reset results
        
    def analyze_image(self):
        try:
            if self.captured_frame is None:
                self.error.emit("No image captured")
                return
                
            self.progress_update.emit("Starting rebar detection...")
            
            # Process with rebar model
            self.detect_rebar(self.captured_frame)
            
        except Exception as e:
            self.error.emit(f"Analysis error: {str(e)}")
            print(traceback.format_exc())
    
    def detect_rebar(self, frame):
        """First detect if there is a rebar in the image"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        height, width = frame_rgb.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            print(f"Resized image for analysis: {width}x{height} -> {new_width}x{new_height}")
        
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run rebar detection model
        self.progress_update.emit("Running rebar detection model...")
        with torch.no_grad():
            outputs = self.rebar_model([inputs])[0]
        
        # Check if any rebars were detected
        if len(outputs["instances"]) == 0:
            self.progress_update.emit("No rebar detected in the image")
            
            no_rebar_filename = os.path.join(self.current_result_dir, 'no_rebar_detected.jpg')
            cv2.imwrite(no_rebar_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Store the result image
            self.result_image = frame_rgb
            
            # Update API data even if no rebar was detected
            self.update_api_data()
            
            # Send results
            results = {
                "status": "complete",
                "message": "No rebar detected",
                "image": frame_rgb,
                "sections": []
            }
            self.analysis_complete.emit(results)
            return
        
        # Get the highest-scoring rebar detection
        instances = outputs["instances"].to("cpu")
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        
        # Get the best rebar detection
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_box = boxes[best_idx].astype(int)
        
        
        # Draw the detected rebar
        rebar_image = frame_rgb.copy()
        x1, y1, x2, y2 = best_box
        cv2.rectangle(rebar_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rebar_image, f"Rebar: {best_score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save rebar detection result
        rebar_filename = os.path.join(self.current_result_dir, 'rebar_detected.jpg')
        cv2.imwrite(rebar_filename, cv2.cvtColor(rebar_image, cv2.COLOR_RGB2BGR))
        
        # Now detect sections within the detected rebar region
        self.detect_sections(frame_rgb, best_box)
        
    def detect_sections(self, frame_rgb, rebar_box):
        """Detect rebar sections within the detected rebar"""
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run section detection model
        self.progress_update.emit("Running section detection model...")
        with torch.no_grad():
            outputs = self.section_model([inputs])[0]
        
        # Get the section instances
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            self.progress_update.emit("No rebar sections detected")
            
            # Store the result image
            self.result_image = frame_rgb
            
            # Update API data
            self.update_api_data()
            
            # Send results
            results = {
                "status": "complete",
                "message": "Rebar detected but no sections found",
                "image": frame_rgb,
                "sections": []
            }
            self.analysis_complete.emit(results)
            return
        
        # Get detection details
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        # Create a result image
        result_image = frame_rgb.copy()
        
        # Generate colors for each section
        section_colors = []
        for i in range(len(boxes)):
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(100, 255)
            )
            section_colors.append(color)
        
        # Process each detected section
        self.progress_update.emit(f"Found {len(boxes)} sections")
        
        # List to store text results for each section
        section_text_results = []
        sections_data = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate diameter in pixels
            width_px = x2 - x1
            height_px = y2 - y1
            diameter_px = min(width_px, height_px)
            
            # Convert to real-world diameter in mm
            mm_per_pixel = 0.1  # Placeholder value
            diameter_mm = diameter_px * mm_per_pixel
            
            # Determine section size based on diameter
            size = "small"
            for category, info in self.cement_ratios.items():
                if "diameter_range" in info:
                    min_diam, max_diam = info["diameter_range"]
                    if min_diam <= diameter_mm < max_diam:
                        size = category
                        break
            
            # Get cement mixture ratio
            ratio = self.cement_ratios[size]
            
            # Calculate volume
            length_cm = height_px * 0.1
            width_cm = width_px * 0.1
            height_cm = width_cm
            volume_cc = length_cm * width_cm * height_cm
            
            # Create text result for this section
            section_result = {
                "section_id": i + 1,
                "size_category": size,
                "diameter_mm": round(diameter_mm, 2),
                "confidence": round(score, 3),
                "width_cm": round(width_cm, 2),
                "length_cm": round(length_cm, 2),
                "height_cm": round(height_cm, 2),
                "volume_cc": round(volume_cc, 2),
                "cement_ratio": ratio["cement"],
                "sand_ratio": ratio["sand"],
                "aggregate_ratio": ratio["aggregate"],
                "bbox": [x1, y1, x2, y2]
            }
            section_text_results.append(section_result)
            
            # Store data for display
            sections_data.append(section_result)
            
            # Save section data for CSV
            section_data = {
                "timestamp": self.current_timestamp,
                **section_result
            }
            self.current_results.append(section_data)
            
            # Draw on the result image
            color = section_colors[i]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"S{i+1}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw mask if available
            if masks is not None:
                mask = masks[i]
                mask_colored = np.zeros_like(frame_rgb)
                mask_colored[mask] = color
                
                # Blend the mask with the image
                alpha = 0.4
                mask_region = mask.astype(bool)
                result_image[mask_region] = (
                    result_image[mask_region] * (1 - alpha) + 
                    mask_colored[mask_region] * alpha
                ).astype(np.uint8)
            
            # Update progress
            self.progress_update.emit(f"Processed section {i+1} ({size})")
        
        # Save result image
        result_filename = os.path.join(self.current_result_dir, 'section_result.jpg')
        cv2.imwrite(result_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Save analysis results to CSV
        self.save_results_to_csv()
        
        # Store the result image for API access and display
        self.result_image = result_image
        
        # Update API data
        self.update_api_data()
        
        # Send results
        results = {
            "status": "complete",
            "message": f"Analysis complete. Found {len(boxes)} sections.",
            "image": result_image,
            "sections": sections_data
        }
        self.analysis_complete.emit(results)

        
    def save_results_to_csv(self):
        """Save the current analysis results to a CSV file"""
        if not self.current_results:
            print("No results to save")
            return
        
        try:
            # Create a CSV file in the analysis folder
            filename = os.path.join(self.current_result_dir, 'analysis_data.csv')
            
            # Define CSV headers
            headers = [
                "timestamp", "section_id", "size_category", "diameter_mm", 
                "confidence", "width_cm", "length_cm", "height_cm", "volume_cc",
                "cement_ratio", "sand_ratio", "aggregate_ratio"
            ]
            
            # Write data to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for result in self.current_results:
                    writer.writerow(result)
            
            print(f"Analysis data saved to {filename}")
            
            # Also save a summary text file
            summary_filename = os.path.join(self.current_result_dir, 'summary.txt')
            with open(summary_filename, 'w') as f:
                analysis_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(time.mktime(
                                                time.strptime(self.current_timestamp, '%Y%m%d-%H%M%S'))))
                f.write(f"Analysis: {analysis_time}\n\n")
                
                if not self.current_results:
                    f.write("No rebar sections detected.\n")
                else:
                    f.write(f"Found {len(self.current_results)} rebar sections:\n\n")
                    for result in self.current_results:
                        f.write(f"Section {result['section_id']} ({result['size_category']}):\n")
                        f.write(f"  Diameter: {result['diameter_mm']:.1f}mm\n")
                        f.write(f"  Mix: C:{result['cement_ratio']} S:{result['sand_ratio']} A:{result['aggregate_ratio']}\n\n")
            
            print(f"Summary saved to {summary_filename}")
            
            self.progress_update.emit(f"Results saved to: analysis_{self.current_timestamp}")
                
        except Exception as e:
            print(f"Error saving analysis data: {e}")
            print(traceback.format_exc())
            self.error.emit(f"Error saving data: {str(e)}")
            
    def update_api_data(self):
        """Update the API data after analysis"""
        global latest_analysis
        
        # Only update if we have results
        if not self.current_results:
            # Even if no segments were detected, we should still update the image
            if self.result_image is not None or self.captured_frame is not None:
                # Use result_image if available, otherwise use captured_frame
                frame_to_encode = self.result_image if self.result_image is not None else self.captured_frame
                
                try:
                    # Convert to RGB format if needed
                    if len(frame_to_encode.shape) == 3 and frame_to_encode.shape[2] == 3:
                        # Check if we need to convert from BGR to RGB
                        if self.result_image is None:  # If using captured_frame, it's in BGR
                            frame_rgb = cv2.cvtColor(frame_to_encode, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame_to_encode  # result_image should already be in RGB
                    else:
                        frame_rgb = frame_to_encode  # Use as is
                    
                    # Convert to PIL Image
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # Save to BytesIO
                    img_io = io.BytesIO()
                    pil_img.save(img_io, 'JPEG', quality=90)
                    img_io.seek(0)
                    
                    # Encode to base64
                    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
                    
                    # Update the latest_analysis
                    latest_analysis["image"] = img_b64
                    latest_analysis["timestamp"] = self.current_timestamp
                    latest_analysis["segments"] = []
                    latest_analysis["total_volume"] = 0
                    
                    # Set image path
                    if self.result_image is not None:
                        latest_analysis["image_path"] = os.path.join(self.current_result_dir, 'section_result.jpg')
                    else:
                        latest_analysis["image_path"] = os.path.join(self.current_result_dir, 'original_image.jpg')
                    
                    print("API data updated with image only (no segments detected)")
                except Exception as e:
                    print(f"Error encoding image for API when no segments detected: {e}")
                    print(traceback.format_exc())
            return
        
        # Convert the current results to the API format
        segments = []
        total_volume = 0
        
        for result in self.current_results:
            segment = {
                "section_id": result["section_id"],
                "size_category": result.get("size_category", "unknown"),
                "diameter_mm": result.get("diameter_mm", 0),
                "confidence": result.get("confidence", 0.9),
                "width_cm": result.get("width_cm", 0),
                "length_cm": result.get("length_cm", 0),
                "height_cm": result.get("height_cm", 0),
                "volume_cc": result.get("volume_cc", 0),
                "bbox": result.get("bbox", [0, 0, 0, 0])
            }
            segments.append(segment)
            total_volume += segment["volume_cc"]
        
        # Convert result image to base64 if available
        if self.result_image is not None:
            try:
                # Convert BGR to RGB if needed (for compatibility)
                image_to_encode = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(image_to_encode)
                
                # Save to BytesIO with higher quality
                img_io = io.BytesIO()
                pil_img.save(img_io, 'JPEG', quality=95)
                img_io.seek(0)
                
                # Encode to base64
                img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
                
                latest_analysis["image"] = img_b64
            except Exception as e:
                print(f"Error converting image for API: {e}")
                print(traceback.format_exc())
                
                # Try with the original captured frame as fallback
                if self.captured_frame is not None:
                    try:
                        frame_rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        img_io = io.BytesIO()
                        pil_img.save(img_io, 'JPEG')
                        img_io.seek(0)
                        img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
                        latest_analysis["image"] = img_b64
                        print("Used original captured frame as fallback for API")
                    except Exception as e2:
                        print(f"Error using fallback image for API: {e2}")
        
        latest_analysis["timestamp"] = self.current_timestamp
        latest_analysis["segments"] = segments
        latest_analysis["total_volume"] = total_volume
        latest_analysis["image_path"] = os.path.join(self.current_result_dir, 'section_result.jpg')
        
        print("API data updated with latest analysis results")
    
    # Main window class
class RebarAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.captured_frame = None
        self.result_image = None
        self.current_results = []
        self.is_processing = False
        self.camera_paused = False
        
        # Create results directory if it doesn't exist
        self.results_dir = "analysis_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Define colors for UI
        self.colors = {
            'primary': '#3498db',      # Blue
            'accent': '#27ae60',       # Green
            'warning': '#e74c3c',      # Red
            'bg': '#f5f5f5',           # Light background
            'dark': '#2c3e50',         # Dark blue/gray
            'text': '#34495e',         # Dark text
            'light_text': '#ffffff'    # Light text
        }
        
        # Set up the UI
        self.setup_ui()
        
        # Start worker threads
        self.setup_workers()
    
    def setup_ui(self):
        # Set window properties
        self.setWindowTitle("Rebar Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.showMaximized()  # Show maximized for better viewing
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Rebar Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Create horizontal splitter for main panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel (camera view)
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(self.left_panel)
        
        # Camera view title
        camera_title = QLabel("Camera")
        camera_title.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(camera_title)
        
        # Camera view area
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(600, 500)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.camera_view.setText("Camera initializing...")
        left_layout.addWidget(self.camera_view, alignment=Qt.AlignCenter)
        
        # Right panel (controls and results)
        self.right_panel = QFrame()
        self.right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(self.right_panel)
        
        # Controls section
        controls_group = QFrame()
        controls_layout = QVBoxLayout(controls_group)
        
        controls_title = QLabel("Controls")
        controls_title.setFont(QFont("Arial", 12, QFont.Bold))
        controls_layout.addWidget(controls_title)
        
        # Capture button
        self.capture_btn = QPushButton("Capture & Analyze")
        self.capture_btn.setFont(QFont("Arial", 11))
        self.capture_btn.setStyleSheet(f"background-color: {self.colors['accent']}; color: white; padding: 10px;")
        self.capture_btn.clicked.connect(self.capture_image)
        controls_layout.addWidget(self.capture_btn)
        
        # Return to camera button (initially hidden)
        self.return_btn = QPushButton("Return to Camera")
        self.return_btn.setFont(QFont("Arial", 11))
        self.return_btn.setStyleSheet(f"background-color: {self.colors['primary']}; color: white; padding: 10px;")
        self.return_btn.clicked.connect(self.resume_camera)
        self.return_btn.hide()
        controls_layout.addWidget(self.return_btn)
        
        right_layout.addWidget(controls_group)
        
        # Results section
        results_group = QFrame()
        results_layout = QVBoxLayout(results_group)
        
        results_title = QLabel("Results")
        results_title.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(results_title)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Arial", 11))
        self.results_text.setStyleSheet("background-color: white; border: 1px solid lightgray;")
        self.results_text.setText("Press 'Capture & Analyze' to begin.")
        results_layout.addWidget(self.results_text)
        
        right_layout.addWidget(results_group)
        
        # Add panels to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([600, 200])  # Set initial sizes
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Initializing...")
        
        # Progress indicator in status bar
        self.progress_label = QLabel()
        self.statusBar.addPermanentWidget(self.progress_label)
    
    def setup_workers(self):
        # Start by loading models
        self.statusBar.showMessage("Loading models...")
        
        # Create thread for model loading
        self.init_thread = QThread()
        self.init_worker = InitWorker()
        self.init_worker.moveToThread(self.init_thread)
        
        # Connect signals
        self.init_thread.started.connect(self.init_worker.load_models)
        self.init_worker.models_loaded.connect(self.on_models_loaded)
        self.init_worker.progress.connect(self.update_status)
        self.init_worker.error.connect(self.show_error)
        
        # Start the thread
        self.init_thread.start()
    
    def on_models_loaded(self, models_data):
        """Called when models are successfully loaded"""
        # Store model references
        self.rebar_model = models_data["rebar_model"]
        self.section_model = models_data["section_model"]
        self.rebar_cfg = models_data["rebar_cfg"]
        self.section_cfg = models_data["section_cfg"]
        self.cement_ratios = models_data["cement_ratios"]
        
        # Update status
        self.update_status("Models loaded successfully")
        
        # Initialize camera worker
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        
        # Connect camera signals
        self.camera_thread.started.connect(self.camera_worker.capture_continuously)
        self.camera_worker.frame_ready.connect(self.update_camera_view)
        self.camera_worker.error.connect(self.show_error)
        
        # Initialize camera
        if self.camera_worker.start_camera():
            # Start the camera thread
            self.camera_thread.start()
            self.update_status("Camera ready")
        else:
            self.show_error("Failed to initialize camera")
        
        # Create analysis worker (but don't start yet)
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(
            self.rebar_model, 
            self.section_model, 
            self.rebar_cfg,
            self.section_cfg,
            self.cement_ratios
        )
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        # Connect analysis signals
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.progress_update.connect(self.update_status)
        self.analysis_worker.error.connect(self.show_error)
        
        # Enable capture button
        self.capture_btn.setEnabled(True)
    
    @pyqtSlot(np.ndarray)
    def update_camera_view(self, frame):
        """Update the camera view with the latest frame"""
        if self.camera_paused:
            return
            
        # Convert frame to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Resize to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.camera_view.width(), self.camera_view.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap to the label
        self.camera_view.setPixmap(pixmap)
    
    def capture_image(self):
        """Capture an image and analyze it"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.capture_btn.setEnabled(False)
        self.update_status("Capturing image...")
        
        # Clear previous results
        self.results_text.clear()
        self.results_text.append("Capturing image...")
        
        # Generate timestamp for this analysis session
        self.current_timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create a unique folder for this analysis session
        self.current_result_dir = os.path.join(self.results_dir, f"analysis_{self.current_timestamp}")
        os.makedirs(self.current_result_dir, exist_ok=True)
        
        # Capture a single frame
        frame = self.camera_worker.capture_single_frame()
        
        if frame is None:
            self.show_error("Failed to capture image")
            self.is_processing = False
            self.capture_btn.setEnabled(True)
            return
            
        # Store the captured frame
        self.captured_frame = frame
        
        # Save original image in the analysis folder
        original_filename = os.path.join(self.current_result_dir, 'original_image.jpg')
        cv2.imwrite(original_filename, self.captured_frame)
        
        # Set up analysis
        self.analysis_worker.set_captured_frame(frame, self.current_timestamp, self.current_result_dir)
        
        # Switch to analysis thread
        if not self.analysis_thread.isRunning():
            self.analysis_thread.start()
        else:
            # Just call the analyze method
            QTimer.singleShot(0, self.analysis_worker.analyze_image)
    
    @pyqtSlot(dict)
    def on_analysis_complete(self, results):
        """Called when analysis is complete"""
        # Display result image
        if results["image"] is not None:
            # Convert to QImage and display
            rgb_image = cv2.cvtColor(results["image"], cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.camera_view.width(), self.camera_view.height(), 
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set the pixmap to the label
            self.camera_view.setPixmap(pixmap)
            
            # Pause camera view updates
            self.camera_paused = True
        
        # Display results in text area
        self.results_text.clear()
        self.results_text.append(results["message"])
        
        if results["sections"]:
            self.results_text.append(f"\nFound {len(results['sections'])} sections:")
            total_volume = 0
            
            for section in results["sections"]:
                self.results_text.append(f"\nSection {section['section_id']} ({section['size_category']}):")
                self.results_text.append(f"  Diameter: {section['diameter_mm']:.1f}mm")
                self.results_text.append(f"  Size: {section['width_cm']:.1f}cm × {section['length_cm']:.1f}cm × {section['height_cm']:.1f}cm")
                self.results_text.append(f"  Volume: {section['volume_cc']:.2f}cc")
                self.results_text.append(f"  Mix: C:{section['cement_ratio']}, S:{section['sand_ratio']}, A:{section['aggregate_ratio']}")
                
                total_volume += section['volume_cc']
                
            self.results_text.append(f"\nTotal Volume: {total_volume:.2f}cc")
        
        # Update status
        self.update_status("Analysis complete")
        
        # Enable return to camera button and hide capture button
        self.capture_btn.hide()
        self.return_btn.show()
        
        # Reset processing flag
        self.is_processing = False

    def resume_camera(self):
        """Resume camera preview"""
        # Resume camera updates
        self.camera_paused = False
        
        # Hide return button and show capture button
        self.return_btn.hide()
        self.capture_btn.show()
        self.capture_btn.setEnabled(True)
        
        # Update status
        self.update_status("Ready")
    
    def update_status(self, message):
        """Update status bar with message"""
        self.statusBar.showMessage(message)
        print(message)  # Also print to console
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        print(f"ERROR: {message}")
        self.statusBar.showMessage(f"Error: {message}")
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Stop threads
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_worker.stop()
            self.camera_thread.quit()
            self.camera_thread.wait()
        
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            
        if hasattr(self, 'init_thread') and self.init_thread.isRunning():
            self.init_thread.quit()
            self.init_thread.wait()
        
        # Accept the close event
        event.accept()
    
# API routes for the Flask server
@api_app.route('/')
def home():
    return "RebarVista API is running!"

@api_app.route('/api/status')
def status():
    """Return the API status"""
    return jsonify({
        "status": "online",
        "camera_available": True,
        "has_results": latest_analysis["timestamp"] is not None
    })

@api_app.route('/api/latest')
def get_latest():
    """Return the latest analysis results (without image)"""
    if latest_analysis["timestamp"] is None:
        return jsonify({
            "timestamp": None,
            "segments": [],
            "total_volume": 0,
            "image_available": False
        })
    
    return jsonify({
        "timestamp": latest_analysis["timestamp"],
        "segments": latest_analysis["segments"],
        "total_volume": latest_analysis["total_volume"],
        "image_available": latest_analysis["image"] is not None
    })

@api_app.route('/api/latest_image')
def get_latest_image():
    """Return the latest analysis image"""
    if latest_analysis["image"] is None:
        return jsonify({"error": "No image available"}), 404
    
    return jsonify({
        "image": latest_analysis["image"]
    })

@api_app.route('/api/download_result_image')
def download_result_image():
    """Send the actual result image file for direct download"""
    if latest_analysis["image_path"] is None or not os.path.exists(latest_analysis["image_path"]):
        return jsonify({"error": "No image file available"}), 404
    
    # Send the file directly
    try:
        directory = os.path.dirname(latest_analysis["image_path"])
        filename = os.path.basename(latest_analysis["image_path"])
        return send_file(latest_analysis["image_path"], 
                         mimetype='image/jpeg',
                         as_attachment=True,
                         download_name=filename)
    except Exception as e:
        print(f"Error sending image file: {e}")
        return jsonify({"error": f"Failed to send image: {str(e)}"}), 500

@api_app.route('/api/capture', methods=["POST"])
def trigger_capture():
    """Trigger a new capture and analysis"""
    try:
        # Use global app_instance to trigger a capture
        if app_instance is not None:
            app_instance.capture_image()
            return jsonify({"message": "Capture triggered successfully"})
        else:
            return jsonify({"error": "Application instance not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_app.route('/api/config', methods=["GET"])
def get_config():
    """Return the current configuration"""
    if app_instance is not None and hasattr(app_instance, 'rebar_cfg'):
        threshold = app_instance.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    else:
        threshold = 0.7
        
    return jsonify({
        "detection_threshold": threshold,
        "camera_enabled": True
    })

@api_app.route('/api/config', methods=["POST"])
def update_config():
    """Update the configuration"""
    try:
        config_data = request.json
        
        if app_instance is not None and hasattr(app_instance, 'rebar_cfg') and "detection_threshold" in config_data:
            # Update threshold
            app_instance.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config_data["detection_threshold"])
            print(f"Updated detection threshold to {config_data['detection_threshold']}")
        
        return jsonify({"message": "Configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_api_server():
    """Start the API server in a separate thread"""
    api_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Main function
def main():
    global app_instance
    
    # Start the API server in a separate thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    print("API server started on port 5000")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Error handling for the entire application
    try:
        # Create main window
        app_instance = RebarAnalysisWindow()
        app_instance.show()
        
        # Run the app
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {e}")
        print(traceback.format_exc())
        QMessageBox.critical(None, "Critical Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()