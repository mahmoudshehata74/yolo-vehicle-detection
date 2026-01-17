# Real-Time Truck Detection (YOLOv11 / YOLOv12) – C++

## Overview
This project implements a **real-time object detector in C++** that detects **trucks**
(and optionally other vehicles) using **YOLOv11/YOLOv12**, accelerated with **GPU (CUDA)**
via **ONNX Runtime**.

The system supports video files, camera input, and RTSP streams, and displays
bounding boxes with confidence scores and FPS in real time.

---

## Features
- Real-time object detection in C++
- YOLOv11 / YOLOv12 detection model (ONNX)
- GPU-accelerated inference using ONNX Runtime (CUDA)
- Truck detection (COCO class id = 7)
- Optional detection of other vehicles (car, bus)
- Bounding boxes with confidence scores
- Non-Maximum Suppression (NMS)
- FPS overlay (instantaneous + moving average)
- Video / Camera / RTSP input via CLI
- OpenCV visualization window

---

## Model
- Model: YOLOv11n (Detection)
- Format: ONNX
- Conversion pipeline:
  PyTorch (.pt) → ONNX → ONNX Runtime (CUDA)

The ONNX models are included in this repository for ease of testing and evaluation.

- `models/yolo11n.onnx`       (Detection)
- `models/yolo11n-seg.onnx`   (Segmentation)

These models were converted using Ultralytics YOLO export.
---

## Build

### Requirements
- C++17
- OpenCV
- ONNX Runtime (GPU build)
- CUDA Toolkit + cuDNN
- NVIDIA GPU (driver installed)

### Build (CMake + Visual Studio)
Open the project folder in Visual Studio (CMake) and build the `task1_detector` target.

---

## Run

### Video file
```bash
task1_detector.exe demo.mp4   ( task1_detector.exe "C:\projects\yolo_tasks_cpp\demo.mp4 )

Camera
task1_detector.exe 0
RTSP   
task1_detector.exe rtsp://username:password@ip:port/stream


### Output

- Live video window with:
		- Bounding boxes
		- Confidence scores
		- FPS overlay

### Notes
	- Truck-only filtering is implemented (COCO class id = 7).
	- For demonstration clarity, the demo may include detection of trucks and other vehicles (car, bus).
	- Confidence/NMS thresholds can be tuned based on the scene and video quality.

### Demo
	- See demo.mp4 (or demo_out.mp4 if video saving is enabled) for a real-time demonstration with visible FPS.

### Author
Mahmoud Shehata Ahmed
