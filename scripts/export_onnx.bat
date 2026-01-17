@echo off
set ROOT=%~dp0..
cd /d %ROOT%

python -m pip install -U ultralytics onnx onnxruntime-gpu onnxsim

yolo export model=yolov11n.pt format=onnx opset=12 simplify=True imgsz=640

if exist yolov11n.onnx (
  move /Y yolov11n.onnx models\
  echo DONE: models\yolov11n.onnx created
) else (
  echo ERROR: ONNX not created. Check output above.
)

pause