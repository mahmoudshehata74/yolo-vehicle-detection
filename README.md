# Real-Time Vehicle Detection & Segmentation (YOLOv11) – C++

## Overview
A real-time C++ application for **vehicle detection and instance segmentation**
using **YOLOv11**, accelerated with **GPU (CUDA)** via **ONNX Runtime**.

Supports video files, camera input, and RTSP streams with real-time visualization.

---

## Tasks

**Task 1 – Detection**
- Vehicle detection (Car, Bus, Truck)
- Bounding boxes with confidence
- FPS overlay (instant + average)

**Task 2 – Segmentation**
- Pixel-perfect instance segmentation
- Transparent masks (adjustable alpha)
- Supports vehicles and persons

---

## Features
- Real-time C++ inference
- YOLOv11 Detection & Segmentation
- ONNX Runtime (CUDA)
- Non-Maximum Suppression (NMS)
- Video / Camera / RTSP input
- Video recording (MP4)
- Screenshot capture

---

## Project Structure
yolo_tasks_cpp/
├── models/ # ONNX models
├── src/ # detector, segmenter, utils
├── CMakeLists.txt
├── README.md
├── demo_detection.mp4
└── demo_segmentation.mp4


---

## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

---

## Run
Detection
task1_detector.exe video.mp4

Segmentation
task2_segmenter.exe video.mp4

## Performance
Detection: ~25–30 FPS
Segmentation: ~30–35 FPS
(Tested on NVIDIA RTX GPU)

## Author
Mahmoud Shehata Ahmed
