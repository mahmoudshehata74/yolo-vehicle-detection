from ultralytics import YOLO
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)

model = YOLO("yolov11n.pt")  # هيحمله تلقائيًا لو مش موجود
out_path = model.export(format="onnx", opset=12, imgsz=640, simplify=False)

out_path = Path(out_path)
target = models_dir / "yolov11n.onnx"
shutil.copy2(out_path, target)

print("DONE:", target)