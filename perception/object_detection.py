import pickle
import json
import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

DATA_ROOT = r"D:\cmp_fourth_year\GP\tayseer\clevr_extracted"
SHAPE_MAP   = {1: "sphere", 2: "cylinder", 3: "cube"}
CLASS_MAP   = {"sphere": 1, "cylinder": 2, "cube": 3}
N_CLUSTERS = 10
HOG_RESIZE = (64, 64)
HOG_ORIENT  = 9
HOG_PPC     = (8, 8)
HOG_CPB     = (2, 2)
IOU_THRESHOLD = 0.5
REGRESSION_THRESHOLD = 0.9
NEGATIVE_SAMPLES_PER_IMAGE = 100
SLIDING_STEP_RATIO    = 0.1
MODEL_PATH= "model.pkl"

def load_mask(mask_dir, obj_id):
    mask_path = os.path.join(mask_dir, f"obj{obj_id:02d}.png")
    if not os.path.exists(mask_path):
        return None
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask

def mask_to_bbox(mask):
    rows, cols = np.where(mask > 0)
    if len(rows) < 50:
        return None

    x1 = int(cols.min())
    y1 = int(rows.min())
    x2 = int(cols.max())
    y2 = int(rows.max())
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None
    return (x1, y1, x2, y2)

def get_annotation(record):
    masks_dir = os.path.join(DATA_ROOT,"masks")
    record_mask_dir = os.path.join(masks_dir, f"{record['record_index']:05d}")
    if not os.path.exists(record_mask_dir):
        return []
    annotations = []
    for obj_id in range(10):
        if record["visibility"][obj_id] < 0.5:
            continue
        shape_id = ord(record["shape"][obj_id])
        if shape_id not in SHAPE_MAP:
            continue
        mask=load_mask(record_mask_dir,obj_id)
        if mask is None:
            continue
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        annotations.append({"bbox": bbox, "class": SHAPE_MAP[shape_id]})
    return annotations



def load_data(maximum_num):
    meta_data = json.load(open(os.path.join(DATA_ROOT, "metadata.json"), "r"))
    imgs_dir = os.path.join(DATA_ROOT, "images")
    
    images = []

    for record in meta_data[:maximum_num]:
        img_path = os.path.join(imgs_dir, f"{record['record_index']:05d}.png")
        if not os.path.exists(img_path):
            continue
        annotations=get_annotation(record)
        if not annotations:
            continue
        images.append({"img_path": img_path, "annotations": annotations})
    return images

def show_image(data_item):
    image = cv2.imread(data_item["img_path"])
    for ann in data_item["annotations"]:
        x1, y1, x2, y2 = ann["bbox"]
        label = ann["class"]
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_window_sizes(train_samples):
    sizes = []
    for sample in train_samples:
        for ann in sample["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            w = x2 - x1
            h = y2 - y1
            sizes.append((w, h))
    sizes = np.array(sizes)
    num_clusters = min(N_CLUSTERS, len(sizes))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(sizes)
    centers = kmeans.cluster_centers_
    window_sizes = []
    for w,h in centers:
        w=max(8, int(w//8)*8)
        h=max(8, int(h//8)*8)
        window_sizes.append((w,h))
    windows = list(set(window_sizes))
    return windows

def extract_hog(patch):
    f = hog(
        patch,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        feature_vector=True
    )
    return f.astype(np.float32)

def compute_iou(boxA, boxB):
    XA = max(boxA[0], boxB[0])
    YA = max(boxA[1], boxB[1])
    XB = min(boxA[2], boxB[2])
    YB = min(boxA[3], boxB[3])
    interArea = max(0, XB - XA) * max(0, YB - YA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def box_to_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h

def compute_regression_targets(gt_box, win_box):
    gt_cx, gt_cy, gt_w, gt_h = box_to_center(gt_box)
    win_cx, win_cy, win_w, win_h = box_to_center(win_box)
    dx = (gt_cx - win_cx) / win_w
    dy = (gt_cy - win_cy) / win_h
    dw = np.log(gt_w / win_w)
    dh = np.log(gt_h / win_h)
    return np.array([dx, dy, dw, dh], dtype=np.float32)

def apply_regression(win_box, reg):
    win_cx, win_cy, win_w, win_h = box_to_center(win_box)
    dx, dy, dw, dh = reg
    gt_cx = win_cx + dx * win_w
    gt_cy = win_cy + dy * win_h
    gt_w = win_w * np.exp(dw)
    gt_h = win_h * np.exp(dh)
    x1 = gt_cx - gt_w / 2
    y1 = gt_cy - gt_h / 2
    x2 = gt_cx + gt_w / 2
    y2 = gt_cy + gt_h / 2
    return x1, y1, x2, y2

def build_training_data(train_samples, window_sizes):
    X, y_cls, y_reg, reg_mask = [], [], [], []
    for sample in train_samples:
        img = cv2.imread(sample["img_path"], cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for ann in sample["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            label = ann["class"]
            class_idx= CLASS_MAP[label]
            patch = gray_img[y1:y2, x1:x2]
            patch = cv2.resize(patch, HOG_RESIZE)
            X.append(extract_hog(patch))
            y_cls.append(class_idx)
            y_reg.append(np.zeros(4, dtype=np.float32))
            reg_mask.append(1) 
            gt_box = (x1, y1, x2, y2)
            gt_w = x2 - x1
            gt_h = y2 - y1
            for jitter_scale in [0.15, 0.25, 0.35, 0.45]:
                for _ in range(6):
                    jx= int(np.random.uniform(-gt_w * jitter_scale, gt_w * jitter_scale))
                    jy= int(np.random.uniform(-gt_h * jitter_scale, gt_h * jitter_scale))
                    scale = np.random.uniform(0.8, 1.2)
                    w = max(8, int(gt_w * scale))
                    h = max(8, int(gt_h * scale))
                    wx1 = max(0, x1 + jx)
                    wy1 = max(0, y1 + jy)
                    wx2 = min(img.shape[1], wx1 + w)
                    wy2 = min(img.shape[0], wy1 + h)
                    if wx2 <= wx1 or wy2 <= wy1:
                        continue
                    win_box = (wx1, wy1, wx2, wy2)
                    iou = compute_iou(gt_box, win_box)
                    if iou < REGRESSION_THRESHOLD:
                        continue
                    patch = gray_img[wy1:wy2, wx1:wx2]
                    patch = cv2.resize(patch, HOG_RESIZE)
                    X.append(extract_hog(patch))
                    y_cls.append(class_idx)
                    y_reg.append(compute_regression_targets(gt_box, win_box))
                    reg_mask.append(1)
        neg_added=0
        attempts=0
        while neg_added<NEGATIVE_SAMPLES_PER_IMAGE and attempts<NEGATIVE_SAMPLES_PER_IMAGE*10:
            attempts+=1
            w,h=window_sizes[np.random.randint(len(window_sizes))]
            x1 = np.random.randint(0, img.shape[1] - w)
            y1 = np.random.randint(0, img.shape[0] - h)
            x2 = x1 + w
            y2 = y1 + h
            win_box = (x1, y1, x2, y2)
            if any(compute_iou(win_box, ann["bbox"]) > IOU_THRESHOLD for ann in sample["annotations"]):
                continue
            patch = gray_img[y1:y2, x1:x2]
            patch = cv2.resize(patch, HOG_RESIZE)
            X.append(extract_hog(patch))
            y_cls.append(-1)
            y_reg.append(np.zeros(4, dtype=np.float32))
            reg_mask.append(0)
            neg_added+=1
    return np.array(X), np.array(y_cls), np.array(y_reg), np.array(reg_mask)


def train_classifier(X_train, y_train):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)
    return clf

def train_regressor(X_train, y_reg_train, reg_mask_train):
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=0.1))
    ])
    reg.fit(X_train[reg_mask_train == 1], y_reg_train[reg_mask_train == 1])
    return reg
def evaluate_classifier(clf,X_train,y_train, X_val, y_val):
    train_preds = clf.predict(X_train)
    val_preds   = clf.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc   = accuracy_score(y_val, val_preds)
    print("\n--- Classifier Evaluation ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}")
    label_names = ["background","sphere", "cylinder", "cube"]
    all_labels = sorted(set(y_train) | set(y_val))
    print(all_labels)
    label_map  = {-1: "background", 1: "sphere", 2: "cylinder", 3: "cube"}
    print("\nTrain Classification Report:")
    print(classification_report(y_train, train_preds, labels=all_labels, target_names=label_names, zero_division=0))
    print("Val Classification Report:")
    print(classification_report(y_val, val_preds, labels=all_labels, target_names=label_names, zero_division=0))

def evaluate_regressor(reg, X_train, y_reg_train, reg_mask_train, X_val, y_reg_val, reg_mask_val):
    X_train_pos   = X_train[reg_mask_train == 1]
    y_train_pos   = y_reg_train[reg_mask_train == 1]
    X_val_pos     = X_val[reg_mask_val == 1]
    y_val_pos     = y_reg_val[reg_mask_val == 1]
    train_preds   = reg.predict(X_train_pos)
    val_preds     = reg.predict(X_val_pos)
    train_mse = mean_squared_error(y_train_pos, train_preds)
    val_mse   = mean_squared_error(y_val_pos,   val_preds)
    train_r2  = r2_score(y_train_pos, train_preds)
    val_r2    = r2_score(y_val_pos,   val_preds)
    print("\n--- Regressor Evaluation ---")
    print(f"Train MSE: {train_mse:.6f}  |  Train R2: {train_r2:.4f}")
    print(f"Val   MSE: {val_mse:.6f}  |  Val   R2: {val_r2:.4f}")
 
def save_models(clf, reg, window_sizes, model_path):
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": clf, "regressor": reg,"window_sizes": window_sizes}, f)
    print(f"\nModels saved to {model_path}")

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter  = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        denom  = areas[i] + areas[order[1:]] - inter
        ious   = np.where(denom > 0, inter / denom, 0)
        order  = order[1:][ious <= iou_thresh]
    return keep

def detect(img_path, clf, reg, window_sizes):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_boxes, all_scores, all_classes = [], [], []
    for w,h in window_sizes:
        step_x = max(1, int(w * SLIDING_STEP_RATIO))
        step_y = max(1, int(h * SLIDING_STEP_RATIO))
        for y in range(0, img.shape[0] - h + 1, step_y):
            for x in range(0, img.shape[1] - w + 1, step_x):
                patch = gray_img[y:y+h, x:x+w]
                patch_resized = cv2.resize(patch, HOG_RESIZE)
                features = extract_hog(patch_resized).reshape(1, -1)
                cls_score = clf.predict_proba(features)[0]
                pred_class_idx = np.argmax(cls_score)
                pred_class = clf.classes_[pred_class_idx]
                if pred_class == -1:
                    continue
                if cls_score[pred_class_idx] < 0.95:
                    continue
                reg_pred = reg.predict(features)[0]
                win_box = (x, y, x+w, y+h)
                refined_box = apply_regression(win_box, reg_pred)
                all_boxes.append(refined_box)
                all_scores.append(cls_score[pred_class_idx])
                all_classes.append(pred_class)
    detected = []
    keep_indices = nms(all_boxes, all_scores, IOU_THRESHOLD)
    for idx in keep_indices:
        detected.append({
            "bbox": all_boxes[idx],
            "score": all_scores[idx],
            "class": all_classes[idx]
        })
    return detected

def evaluate_detections(detections, annotations):
    gt_boxes = [ann["bbox"] for ann in annotations]
    gt_classes = [CLASS_MAP[ann["class"]] for ann in annotations]
    matched_gt = set()
    tp, fp, fn = 0, 0, 0
    for det in detections:
        det_box = det["bbox"]
        det_class = det["class"]
        best_iou = 0
        best_gt_idx = -1
        for idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if idx in matched_gt:
                continue
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= IOU_THRESHOLD and det_class == gt_classes[best_gt_idx]:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score



if __name__ == "__main__":
    np.random.seed(42)
 
    samples = load_data(200)
 
    print(f"Loaded {len(samples)} images with annotations.")
 
    if len(samples) < 2:
        print("Too few samples")
        exit()
 
    np.random.shuffle(samples)
 
    split_idx = int(0.8 * len(samples))
 
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:]
 
    window_sizes = make_window_sizes(train_samples)
 
    print("Window sizes:", window_sizes)
 
    X_train, y_train, y_reg_train, reg_mask_train = build_training_data(
        train_samples,
        window_sizes
    )
 
    X_val, y_val, y_reg_val, reg_mask_val = build_training_data(
        val_samples,
        window_sizes
    )
 
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
 
    print("Training classifier...")
    clf = train_classifier(X_train, y_train)
 
    print("Training regressor...")
    reg = train_regressor(X_train, y_reg_train, reg_mask_train)
 
    evaluate_classifier(clf, X_train, y_train, X_val, y_val)
    evaluate_regressor(reg, X_train, y_reg_train, reg_mask_train, X_val, y_reg_val, reg_mask_val)
 
    save_models(clf, reg, window_sizes, MODEL_PATH)
 
    precisions = []
    recalls = []
    f1_scores = []
 
    for sample in val_samples:
 
        detections = detect(
            sample["img_path"],
            clf,
            reg,
            window_sizes
        )
 
        precision, recall, f1 = evaluate_detections(
            detections,
            sample["annotations"]
        )
 
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
 
        image = cv2.imread(sample["img_path"])
 
        for det in detections:
 
            x1, y1, x2, y2 = map(int, det["bbox"])
 
            cls_name = SHAPE_MAP[det["class"]]
 
            score = det["score"]
 
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
 
            cv2.putText(
                image,
                f"{cls_name}: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
 
        cv2.imshow("Detections", image)
 
        key = cv2.waitKey(0)
 
        if key == 27:
            break
 
    cv2.destroyAllWindows()
 
    print("\nEvaluation Results")
 
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1 Score:  {np.mean(f1_scores):.4f}")
 