

import argparse
import pickle
import os
import cv2
import numpy as np
from skimage.feature import hog


HOG_RESIZE         = (64, 64)
HOG_ORIENT         = 9
HOG_PPC            = (8, 8)
HOG_CPB            = (2, 2)
IOU_THRESHOLD      = 0.5
SLIDING_STEP_RATIO = 0.1
SHAPE_MAP          = {1: "sphere", 2: "cylinder", 3: "cube"}
COLORS             = {1: (0, 255, 0), 2: (255, 128, 0), 3: (0, 128, 255)}
MODEL_PATH         = "model.pkl"
IMAGE_SIZE         = (240, 320)

def extract_hog(patch):
    return hog(patch, orientations=HOG_ORIENT, pixels_per_cell=HOG_PPC,
               cells_per_block=HOG_CPB, feature_vector=True).astype(np.float32)


def apply_regression(win_box, reg):
    x1, y1, x2, y2 = win_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w,  h  = x2 - x1, y2 - y1
    dx, dy, dw, dh = reg
    cx2 = cx + dx * w
    cy2 = cy + dy * h
    w2  = w  * np.exp(dw)
    h2  = h  * np.exp(dh)
    return cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2


def nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        denom = areas[i] + areas[order[1:]] - inter
        ious  = np.where(denom > 0, inter / denom, 0)
        order = order[1:][ious <= .35]
    return keep


def detect(img_path, clf, reg, window_sizes, conf_thresh=0.95):
    img  = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    H, W = gray.shape[:2]

    all_boxes, all_scores, all_classes = [], [], []

    for w, h in window_sizes:
        step_x = max(1, int(w * SLIDING_STEP_RATIO))
        step_y = max(1, int(h * SLIDING_STEP_RATIO))
        for y in range(0, H - h + 1, step_y):
            for x in range(0, W - w + 1, step_x):
                patch    = cv2.resize(gray[y:y+h, x:x+w], HOG_RESIZE)
                features = extract_hog(patch).reshape(1, -1)

                proba    = clf.predict_proba(features)[0]
                pred_idx = np.argmax(proba)
                cls      = clf.classes_[pred_idx]
                score    = proba[pred_idx]

                if cls == -1 or score < conf_thresh:
                    continue

                refined = apply_regression((x, y, x+w, y+h),
                                           reg.predict(features)[0])
                all_boxes.append(refined)
                all_scores.append(score)
                all_classes.append(cls)

    keep = nms(all_boxes, all_scores, IOU_THRESHOLD)
    return [{"bbox": all_boxes[i], "score": all_scores[i],
             "class": all_classes[i]} for i in keep]


def draw_detections(img_path, detections, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img=cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls   = det["class"]
        color = COLORS.get(cls, (255, 255, 255))
        label = f"{SHAPE_MAP.get(cls, '?')} {det['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(output_path, img)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal object detector")
    parser.add_argument("--image",  required=True,              help="Input image path")
    parser.add_argument("--model",  default=MODEL_PATH,         help="Model pickle path")
    parser.add_argument("--output", default="output.png",       help="Output image path")
    parser.add_argument("--conf",   type=float, default=0.95,   help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf, reg, window_sizes = bundle["classifier"], bundle["regressor"], bundle["window_sizes"]

    dets = detect(args.image, clf, reg, window_sizes, conf_thresh=args.conf)
    draw_detections(args.image, dets, args.output)

    print(f"\nDetected {len(dets)} object(s):")
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        print(f"  {SHAPE_MAP.get(d['class'], '?'):10s}  "
              f"score={d['score']:.3f}  box=({x1},{y1},{x2},{y2})")