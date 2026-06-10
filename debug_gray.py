import os
import cv2
from ultralytics import YOLO

gray_model_path = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\best-gray.pt"
test_dataset_dir = r"C:\Users\sevyl_26tvvn3\OneDrive\Masaüstü\hibrit model\test-dataset-2"

print("Loading gray model...")
model = YOLO(gray_model_path)
print("Model loaded.")

# Let's find images
import glob
images = glob.glob(os.path.join(test_dataset_dir, "*.jpg")) + glob.glob(os.path.join(test_dataset_dir, "*.png"))

print(f"Found {len(images)} images.")

for i, img_path in enumerate(images[:5]):
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Run with different conf values
    for conf in [0.25, 0.1, 0.05, 0.01]:
        res_bgr = model.predict(img_bgr, conf=conf, verbose=False)
        res_gray = model.predict(img_gray_3ch, conf=conf, verbose=False)
        
        boxes_bgr = len(res_bgr[0].boxes) if len(res_bgr) > 0 else 0
        boxes_gray = len(res_gray[0].boxes) if len(res_gray) > 0 else 0
        
        print(f"Img {i} ({os.path.basename(img_path)}), Conf={conf}:")
        print(f"  BGR input detections: {boxes_bgr}")
        print(f"  Gray 3ch input detections: {boxes_gray}")
