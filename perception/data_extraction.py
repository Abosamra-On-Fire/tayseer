import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image





MAX_NUM_ENTITIES = 11  
IMAGE_SIZE       = (240, 320)  

FEATURES = {
    "image":        tf.io.FixedLenFeature(IMAGE_SIZE + (3,),                    tf.string),
    "mask":         tf.io.FixedLenFeature([MAX_NUM_ENTITIES] + list(IMAGE_SIZE) + [1], tf.string),
    "x":            tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.float32),
    "y":            tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.float32),
    "z":            tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.float32),
    "pixel_coords": tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3],                tf.float32),
    "rotation":     tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.float32),
    "size":         tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.string),
    "material":     tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.string),
    "shape":        tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.string),
    "color":        tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.string),
    "visibility":   tf.io.FixedLenFeature([MAX_NUM_ENTITIES],                   tf.float32),
}


def parse_record(raw: bytes) -> dict:
    return tf.io.parse_single_example(raw, FEATURES)


def decode_image(parsed: dict) -> np.ndarray:
    flat = tf.io.decode_raw(parsed["image"], tf.uint8)
    return flat.numpy().reshape(IMAGE_SIZE + (3,))


def decode_masks(parsed: dict) -> np.ndarray:
    """mask: MAX_NUM_ENTITIES slots of raw uint8 → (H, W, MAX_NUM_ENTITIES)."""
    masks = []
    raw_masks = parsed["mask"]
    for i in range(MAX_NUM_ENTITIES):
        flat = tf.io.decode_raw(raw_masks[i], tf.uint8)
        masks.append(flat.numpy().reshape(IMAGE_SIZE + (1,)))
    return np.concatenate(masks, axis=-1)


def extract_metadata(parsed: dict, idx: int) -> dict:
    def to_list(t):
        v = t.numpy()
        return [x.decode("utf-8") if isinstance(x, bytes) else float(x) for x in v]

    return {
        "record_index": idx,
        "color":        to_list(parsed["color"]),
        "shape":        to_list(parsed["shape"]),
        "material":     to_list(parsed["material"]),
        "size":         to_list(parsed["size"]),
        "x":            parsed["x"].numpy().tolist(),
        "y":            parsed["y"].numpy().tolist(),
        "z":            parsed["z"].numpy().tolist(),
        "rotation":     parsed["rotation"].numpy().tolist(),
        "visibility":   parsed["visibility"].numpy().tolist(),
        "pixel_coords": parsed["pixel_coords"].numpy().tolist(),
    }


def main():
    data_path=r"D:\cmp_fourth_year\GP\Object_detection\clevr_with_masks_train.tfrecords"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    extracted_dir=r"D:\cmp_fourth_year\GP\Object_detection\clevr_extracted"
    maximum_records=None
    images_dir = os.path.join(extracted_dir, "images")
    masks_dir  = os.path.join(extracted_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)
    dataset = tf.data.TFRecordDataset(data_path, compression_type="GZIP")

    all_meta = []
    n_ok     = 0
    n_fail   = 0

    for raw in dataset:
        if maximum_records and n_ok >= maximum_records:
            break

        idx = n_ok + n_fail

        try:
            parsed = parse_record(raw.numpy())
        except Exception as e:
            print(f"  [WARN] record {idx} parse failed: {e}")
            n_fail += 1
            continue

        
        try:
            img = decode_image(parsed)
            Image.fromarray(img).save(os.path.join(images_dir, f"{n_ok:05d}.png"))
        except Exception as e:
            print(f"  [WARN] record {idx} image failed: {e}")

        
        try:
            masks = decode_masks(parsed)
            rec_dir = os.path.join(masks_dir, f"{n_ok:05d}")
            os.makedirs(rec_dir, exist_ok=True)
            for obj_i in range(MAX_NUM_ENTITIES):
                m = masks[:, :, obj_i]
                if m.max() <= 1:
                    m = (m * 255).astype(np.uint8)
                Image.fromarray(m).save(os.path.join(rec_dir, f"obj{obj_i:02d}.png"))
        except Exception as e:
            print(f"  [WARN] record {idx} mask failed: {e}")

        
        try:
            all_meta.append(extract_metadata(parsed, n_ok))
        except Exception as e:
            print(f"  [WARN] record {idx} metadata failed: {e}")

        n_ok += 1
        if n_ok % 100 == 0 or n_ok == 1:
            print(f"  Extracted {n_ok} records  (skipped {n_fail})...")

    if all_meta:
        meta_path = os.path.join(extracted_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2)
        print(f"\nMetadata -> {meta_path}")

    print(f"\nDone! Extracted {n_ok} records, skipped {n_fail}.")
    print(f"Images : {images_dir}")
    print(f"Masks  : {masks_dir}  (sub-folder per image, {MAX_NUM_ENTITIES} PNGs each)")


if __name__ == "__main__":
    main()