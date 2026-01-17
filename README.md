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
```
yolo_tasks_cpp/
├── models/ # ONNX models
├── src/ # detector, segmenter, utils
├── CMakeLists.txt
├── README.md
├── demo_detection.mp4
└── demo_segmentation.mp4


The ONNX models are included in this repository for ease of testing and evaluation.

- `models/yolo11n.onnx`       (Detection)
- `models/yolo11n-seg.onnx`   (Segmentation)

These models were converted using Ultralytics YOLO export.
---
```
## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

```
---

## Run
- Detection
```bash 
task1_detector.exe video2.mp4
```
- Segmentation
```bash
task2_segmenter.exe video.mp4
```
---

## Performance

- Detection: ~50–55 FPS.
- Segmentation: ~25–30 FPS.
- Tested on NVIDIA GTX 1650 Ti 

---

## Demo Videos

Recorded demo videos for both tasks are available in the GitHub Releases section:

👉 **https://github.com/mahmoudshehata74/yolo-vehicle-detection/releases/tag/v1.0**

- `demo_detection.mp4` – Real-time vehicle detection with FPS overlay  
- `demo_segmentation.mp4` – Pixel-perfect segmentation with transparent masks

## Author
- Mahmoud Shehata Ahmed

