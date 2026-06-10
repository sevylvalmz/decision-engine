import sys
import os

print(f"Python version: {sys.version}")

try:
    import torch
    print(f"torch is installed (version: {torch.__version__})")
except ImportError:
    print("torch is NOT installed")

try:
    import cv2
    print(f"opencv (cv2) is installed (version: {cv2.__version__})")
except ImportError:
    print("opencv (cv2) is NOT installed")

try:
    import ultralytics
    print(f"ultralytics (YOLO) is installed (version: {ultralytics.__version__})")
except ImportError:
    print("ultralytics is NOT installed")

try:
    import numpy as np
    print(f"numpy is installed (version: {np.__version__})")
except ImportError:
    print("numpy is NOT installed")

# Path to the models
gray_model_path = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\best-gray.pt"
rgb_model_path = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\best-rgb.pt"

print(f"\nChecking models:")
print(f"best-gray.pt exists: {os.path.exists(gray_model_path)}")
print(f"best-rgb.pt exists: {os.path.exists(rgb_model_path)}")

if os.path.exists(gray_model_path):
    print(f"best-gray.pt size: {os.path.getsize(gray_model_path)} bytes")
if os.path.exists(rgb_model_path):
    print(f"best-rgb.pt size: {os.path.getsize(rgb_model_path)} bytes")

# Let's try loading model metadata if torch/ultralytics is available
if 'ultralytics' in sys.modules:
    from ultralytics import YOLO
    print("\nAttempting to load models with Ultralytics YOLO...")
    try:
        if os.path.exists(gray_model_path):
            model_gray = YOLO(gray_model_path)
            print("Loaded best-gray.pt successfully!")
            print(f"Gray Model Names: {model_gray.names}")
            print(f"Gray Model Task: {model_gray.task}")
        if os.path.exists(rgb_model_path):
            model_rgb = YOLO(rgb_model_path)
            print("Loaded best-rgb.pt successfully!")
            print(f"RGB Model Names: {model_rgb.names}")
            print(f"RGB Model Task: {model_rgb.task}")
    except Exception as e:
        print(f"Failed to load via YOLO: {e}")
elif 'torch' in sys.modules:
    print("\nAttempting to load models with PyTorch...")
    try:
        if os.path.exists(gray_model_path):
            checkpoint = torch.load(gray_model_path, map_location='cpu')
            print("Loaded best-gray.pt successfully via torch.load!")
            if isinstance(checkpoint, dict):
                print(f"Keys in checkpoint: {list(checkpoint.keys())}")
                if 'model' in checkpoint:
                    print(f"Model class: {type(checkpoint['model'])}")
        if os.path.exists(rgb_model_path):
            checkpoint = torch.load(rgb_model_path, map_location='cpu')
            print("Loaded best-rgb.pt successfully via torch.load!")
            if isinstance(checkpoint, dict):
                print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Failed to load via torch: {e}")
