import os
import cv2
import glob
from ultralytics import YOLO

gray_model_path = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\best-gray.pt"
rgb_model_path = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\best-rgb.pt"
test_dataset_dir = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\test-dataset-2"

print("Loading models...")
model_gray = YOLO(gray_model_path)
model_rgb = YOLO(rgb_model_path)

images = glob.glob(os.path.join(test_dataset_dir, "*.jpg")) + glob.glob(os.path.join(test_dataset_dir, "*.png"))

print(f"Comparing detections on {len(images)} images (Conf=0.25):")
print("-" * 70)
print(f"{'Image Name':<45} | {'RGB Dets':<8} | {'Gray Dets (on BGR)':<18}")
print("-" * 70)

total_rgb = 0
total_gray = 0

for img_path in images:
    img_name = os.path.basename(img_path)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
        
    res_rgb = model_rgb.predict(img_bgr, conf=0.25, verbose=False)
    res_gray = model_gray.predict(img_bgr, conf=0.25, verbose=False)
    
    dets_rgb = len(res_rgb[0].boxes) if len(res_rgb) > 0 else 0
    dets_gray = len(res_gray[0].boxes) if len(res_gray) > 0 else 0
    
    total_rgb += dets_rgb
    total_gray += dets_gray
    
    print(f"{img_name:<45} | {dets_rgb:<8} | {dets_gray:<18}")

print("-" * 70)
print(f"{'TOTAL':<45} | {total_rgb:<8} | {total_gray:<18}")
