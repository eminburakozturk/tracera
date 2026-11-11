"""Ortamın eğitime hazır olup olmadığını kontrol eder."""
import sys

print(f"Python      : {sys.version.split()[0]}")

import torch
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA hazır  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU         : {p.name} ({p.total_memory / 1024**3:.1f} GB)")
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize(); (x @ x).sum().item()
    print("GPU testi   : OK")
else:
    print("UYARI: GPU görünmüyor. PyTorch CPU sürümü kurulmuş olabilir.")

import ultralytics
print(f"Ultralytics : {ultralytics.__version__}")

for mod in ["mlflow", "onnx", "onnxruntime", "openvino", "pytorch_grad_cam", "gradio", "cv2"]:
    try:
        m = __import__(mod)
        print(f"{mod:<12}: {getattr(m, '__version__', 'OK')}")
    except Exception as e:
        print(f"{mod:<12}: EKSİK ({e.__class__.__name__})")

try:
    import onnxruntime as ort
    print(f"ORT sağlayıcılar: {ort.get_available_providers()}")
except Exception:
    pass

print("\nHer şey OK ise proje başlamaya hazır.")
