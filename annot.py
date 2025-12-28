import json
import os
import cv2
import numpy as np
import shutil
import yaml
from sklearn.model_selection import train_test_split

# ==============================
# Input Paths 
IMAGE_DIR = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\images"
JSON_DIR  = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\COCO"
# Output Path for YOLO Dataset
DATASET_DIR = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\yolo_dataset"
CLASS_NAMES = ["BG", "neutrophil", "eosinophil", "basophil", "monocyte", "lymphocyte"]

CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES) if name != "BG"}

# ==============================
# MASK GENERATION 
# ==============================
def generate_mask_from_bbox(img_bgr, bbox):
   
    x, y, w, h = map(int, bbox)
    H, W = img_bgr.shape[:2]
    
    # Clamp
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    
    if w <= 0 or h <= 0:
        return None
    roi = img_bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour += np.array([x, y]) # Offset
    
    # Simplify contour
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 3:
        return None
        
    # Normalize coordinates (0-1)
    polygon = []
    for point in approx:
        px, py = point[0]
        polygon.append(px / W)
        polygon.append(py / H)
        
    return polygon
# ==============================
# CONVERSION 
# ==============================
def convert_to_yolo():
    # Setup Directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"Found {len(image_files)} images.")
    # Split Data
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"Processing {split} set ({len(files)} images)...")
        
        for img_name in files:
            base = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMAGE_DIR, img_name)
            json_path = os.path.join(JSON_DIR, base + ".json")
            
            if not os.path.exists(json_path):
                continue
                
            # Read Image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Read JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Parse Categories
            categories = {c["id"]: c["name"] for c in data.get("categories", [])}
            
            yolo_lines = []
            
            for ann in data.get("annotations", []):
                cat_name = categories.get(ann["category_id"])
                if cat_name not in CLASS_MAP:
                    continue
                    
                class_id = CLASS_MAP[cat_name] - 1 
                
                bbox = ann.get("bbox")
                if not bbox:
                    continue
                    
                # Generate Polygon
                polygon = generate_mask_from_bbox(img, bbox)
                
                if polygon:
                    # ClassID x1 y1 x2 y2 ...
                    line = f"{class_id} " + " ".join(map(str, polygon))
                    yolo_lines.append(line)
                else:
                    
                    x, y, w, h = map(int, bbox)
                    H, W = img.shape[:2]
                    
                    pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    poly = []
                    for px, py in pts:
                        poly.append(min(max(px,0),W)/W)
                        poly.append(min(max(py,0),H)/H)
                    line = f"{class_id} " + " ".join(map(str, poly))
                    yolo_lines.append(line)
            # Save Label
            if yolo_lines:
                label_path = os.path.join(DATASET_DIR, split, 'labels', base + ".txt")
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
                
                # Copy Image
                dst_img_path = os.path.join(DATASET_DIR, split, 'images', img_name)
                shutil.copy(img_path, dst_img_path)
    # Create dataset.yaml
    yaml_content = {
        'path': DATASET_DIR,
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASS_NAMES) if name != "BG"}
    }
   
    names_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        if name == "BG": continue
        names_dict[i-1] = name
    
    yaml_content['names'] = names_dict
    with open(os.path.join(DATASET_DIR, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Dataset prepared at {DATASET_DIR}")
    print(f"YAML file created at {os.path.join(DATASET_DIR, 'dataset.yaml')}")
if __name__ == "__main__":
    convert_to_yolo()


    #========================================================================================================


from ultralytics import YOLO
import os

# ==============================
DATASET_YAML = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\yolo_dataset\dataset.yaml"
MODEL_NAME = "yolov8n-seg.pt" 
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 32
def train_model():
    # Load model
    model = YOLO(MODEL_NAME)
    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="wbc_segmentation",
        exist_ok=True 
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
if __name__ == "__main__":
    train_model()


    #================================================================================================================


from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# ==============================
# Path to the TRAINED model 
MODEL_PATH = r"C:\Users\babi8\runs\segment\wbc_segmentation\weights\best.pt"
# Input/Output
INPUT_DIR = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\images" # Or a test folder
OUTPUT_DIR = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\yolo_out"
CLASS_NAMES = ["neutrophil", "eosinophil", "basophil", "monocyte", "lymphocyte"]

ALPHA = 0.6
LABEL_ALPHA = 1.0
# ==============================
# COLOR GENERATION 
# ==============================
def get_class_color(class_name):
    colors = {
        "neutrophil": (255, 105, 180),   # Hot Pink
        "eosinophil": (50, 205, 50),     # Lime Green
        "basophil": (138, 43, 226),      # Blue Violet
        "monocyte": (255, 165, 0),       # Orange
        "lymphocyte": (0, 191, 255),     # Deep Sky Blue
    }
    return colors.get(class_name, (200, 200, 200))
# ==============================
# DRAW FUNCTION 
# ==============================
def draw_rounded_rect(img, pt1, pt2, color, thickness, r, alpha):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.circle(overlay, (x1+r, y1+r), r, color, -1)
    cv2.circle(overlay, (x2-r, y1+r), r, color, -1)
    cv2.circle(overlay, (x1+r, y2-r), r, color, -1)
    cv2.circle(overlay, (x2-r, y2-r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
def visualize(img_bgr, results):
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    if results[0].masks is None:
        return img_bgr
    masks = results[0].masks.data.cpu().numpy() # Masks (N, H, W) - usually resized
    boxes = results[0].boxes.data.cpu().numpy() # (N, 6) -> x1, y1, x2, y2, conf, cls
    
    
    polygons = results[0].masks.xy
    
    for i, poly in enumerate(polygons):
        if len(poly) == 0: continue
        
        box = boxes[i]
        cls_id = int(box[5])
        class_name = CLASS_NAMES[cls_id]
        
        color_rgb = get_class_color(class_name)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Draw Mask
        pts = np.array(poly, np.int32)
        cv2.fillPoly(overlay, [pts], color_bgr)
        
        # Draw BBox
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 4)
        
        # Draw Label
        text = class_name
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        
        pad_x, pad_y = 10, 6
        label_x = x1
        label_y = y1 - text_h - 2 * pad_y - 5
        
        if label_y < 0: label_y = y1 + (y2-y1) + 5
        if label_x + text_w + 2*pad_x > w: label_x = w - text_w - 2*pad_x
        
        pt1 = (int(label_x), int(label_y))
        pt2 = (int(label_x + text_w + 2*pad_x), int(label_y + text_h + 2*pad_y))
        
        draw_rounded_rect(img_bgr, pt1, pt2, color_bgr, -1, 10, 1.0)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((label_x + pad_x, label_y + pad_y - 2), text, font=font, fill=(255, 255, 255))
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.addWeighted(overlay, ALPHA, img_bgr, 1 - ALPHA, 0, img_bgr)
    return img_bgr
def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run train.py first!")
        return
    model = YOLO(MODEL_PATH)
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"Found {len(image_files)} images.")
    
    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Inference
        results = model(img, verbose=False)
        
        
        out_img = visualize(img.copy(), results)
        
        
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, out_img)
        print(f"[OK] Saved: {out_path}")
if __name__ == "__main__":
    run_inference()

    #========================================================================================================================


import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ==============================
IMAGE_DIR = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\images"
JSON_DIR  = r"C:\Users\babi8\Downloads\WBC dataset\inputWBCdata\COCO"
MODEL_PATH = r"C:\Users\babi8\runs\segment\wbc_segmentation\weights\best.pt"

CLASS_NAMES = ["neutrophil", "eosinophil", "basophil", "monocyte", "lymphocyte"]
IOU_THRESHOLD = 0.5

# ==============================
# UTILITY FUNCTIONS
# ==============================
def polygon_to_mask(polygon, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

# ==============================
# LOAD GROUND TRUTH
# ==============================
def load_ground_truth(json_path, img_shape):
    h, w = img_shape[:2]
    with open(json_path, "r") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    gt_objects = []

    for ann in data["annotations"]:
        cls_name = categories[ann["category_id"]]
        if cls_name not in CLASS_NAMES:
            continue

        if "segmentation" in ann and ann["segmentation"]:
            poly = ann["segmentation"][0]
            poly = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            mask = polygon_to_mask(poly, h, w)
            gt_objects.append((cls_name, mask))

    return gt_objects

# ==============================
# EVALUATION
# ==============================
def evaluate():
    model = YOLO(MODEL_PATH)

    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGE_DIR, img_name)
        json_path = os.path.join(JSON_DIR, base + ".json")

        if not os.path.exists(json_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_objects = load_ground_truth(json_path, img.shape)
        used_gt = set()

        results = model(img, verbose=False)[0]

        if results.masks is None:
            for cls_name, _ in gt_objects:
                metrics[cls_name]["FN"] += 1
            continue

        pred_masks = results.masks.data.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, pred_mask in enumerate(pred_masks):
            cls_name = CLASS_NAMES[pred_classes[i]]
            pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
            pred_mask = pred_mask > 0.5

            best_iou = 0
            best_j = -1

            for j, (gt_cls, gt_mask) in enumerate(gt_objects):
                if j in used_gt or gt_cls != cls_name:
                    continue

                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= IOU_THRESHOLD:
                metrics[cls_name]["TP"] += 1
                used_gt.add(best_j)
            else:
                metrics[cls_name]["FP"] += 1

        for j, (gt_cls, _) in enumerate(gt_objects):
            if j not in used_gt:
                metrics[gt_cls]["FN"] += 1

    # ==============================
    # METRIC CALCULATION
    # ==============================
    print("\nEvaluation Results:\n")
    total_TP = total_FP = total_FN = 0

    for cls in CLASS_NAMES:
        TP = metrics[cls]["TP"]
        FP = metrics[cls]["FP"]
        FN = metrics[cls]["FN"]

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = TP / (TP + FP + FN + 1e-6)

        total_TP += TP
        total_FP += FP
        total_FN += FN

        print(f"{cls:12s} | "
              f"P: {precision:.4f}  "
              f"R: {recall:.4f}  "
              f"F1: {f1:.4f}  "
              f"Acc: {accuracy:.4f}")

    overall_precision = total_TP / (total_TP + total_FP + 1e-6)
    overall_recall = total_TP / (total_TP + total_FN + 1e-6)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
    overall_accuracy = total_TP / (total_TP + total_FP + total_FN + 1e-6)

    print("\nOverall Metrics:")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall   : {overall_recall:.4f}")
    print(f"F1-score : {overall_f1:.4f}")
    print(f"Accuracy : {overall_accuracy:.4f}")


# ==============================
if __name__ == "__main__":
    evaluate()



