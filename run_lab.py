import hashlib
import json
import platform
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
from PIL import Image
import torch
import ultralytics
from ultralytics import YOLO

ULTRALYTICS_VERSION_PIN = "8.4.145"
CLASSIFICATION_MODEL_FILE = "yolo11n-cls.pt"
DETECTION_MODEL_FILE = "yolo11n.pt"
SEGMENTATION_MODEL_FILE = "yolo11n-seg.pt"

MODEL_ASSETS = {
    "yolo11n-cls.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-cls.pt",
        "sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    },
    "yolo11n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    },
    "yolo11n-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt",
        "sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    },
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("day1_lab_outputs")
VISUAL_DIR = OUTPUT_DIR / "visuals"
IMAGE_DIR = Path("day1_lab_images")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
for directory in (VISUAL_DIR, IMAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_with_sha256(label, url, target, expected_sha256):
    target = Path(target)
    if target.exists() and sha256_file(target) != expected_sha256:
        target.unlink()
    if not target.exists():
        temporary_target = target.with_name(f".{target.name}.download")
        temporary_target.unlink(missing_ok=True)
        try:
            urlretrieve(url, temporary_target)
        except (HTTPError, URLError) as error:
            temporary_target.unlink(missing_ok=True)
            raise RuntimeError(f"Không tải được {label}; chạy lại một lần rồi báo mentor.") from error
        actual_sha256 = sha256_file(temporary_target)
        if actual_sha256 != expected_sha256:
            temporary_target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum sai cho {label}: expected={expected_sha256}, actual={actual_sha256}. "
                "Không load tệp này; hãy báo mentor."
            )
        temporary_target.replace(target)
    actual_sha256 = sha256_file(target)
    return target, actual_sha256

MODEL_SHA256 = {}
for model_file, metadata in MODEL_ASSETS.items():
    _, actual_sha256 = download_with_sha256(
        f"model {model_file}", metadata["url"], model_file, metadata["sha256"]
    )
    MODEL_SHA256[model_file] = actual_sha256
    print(f"PASS model={model_file} sha256={actual_sha256[:12]}…")

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

SAMPLES = {
    "traffic": {
        "coco_image_id": 210273,
        "url": "http://images.cocodataset.org/val2017/000000210273.jpg",
        "sha256": "3ec23de63592c1eef86740fd46fed2cb66170fca02b695d37e6ae379dcee4355",
        "source_title": "Wuhan", "source_creator": "Tauno Tõhk (toehk)",
        "source_url": "https://www.flickr.com/photo.gne?id=5336041838",
        "license_name": "CC BY 2.0", "license_url": "https://creativecommons.org/licenses/by/2.0/",
    },
    "kitchen": {
        "coco_image_id": 397133,
        "url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "sha256": "09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4",
        "source_title": "Kitchen", "source_creator": "Maggie Stephens (Pot Noodle)",
        "source_url": "https://www.flickr.com/photo.gne?id=6255196340",
        "license_name": "CC BY 2.0", "license_url": "https://creativecommons.org/licenses/by/2.0/",
    },
    "dining": {
        "coco_image_id": 166918,
        "url": "http://images.cocodataset.org/val2017/000000166918.jpg",
        "sha256": "a7f8457580a2bb8635ca7acd1b2061ea7977a84418e5e8441a77ba1f861d7e22",
        "source_title": "Big Wine Thing", "source_creator": "WordRidden",
        "source_url": "https://www.flickr.com/photo.gne?id=4745624149",
        "license_name": "CC BY 2.0", "license_url": "https://creativecommons.org/licenses/by/2.0/",
    },
}

for sample_id, metadata in SAMPLES.items():
    target = IMAGE_DIR / f"{sample_id}.jpg"
    _, actual_sha256 = download_with_sha256(
        f"sample {sample_id}", metadata["url"], target, metadata["sha256"]
    )
    print(f"PASS sample={sample_id} coco_id={metadata['coco_image_id']} sha256={actual_sha256[:12]}…")

attribution_lines = [
    "# Image attribution", "",
    "Source images were downloaded from COCO 2017 validation. Generated PNG evidence adds model predictions/charts; these are modified versions.",
    "", "| Sample | Original work / creator | Source | License | Change |",
    "| --- | --- | --- | --- | --- |",
]
for sample_id, metadata in SAMPLES.items():
    attribution_lines.append(
        f"| `{sample_id}` (COCO {metadata['coco_image_id']}) | {metadata['source_title']} / {metadata['source_creator']} | "
        f"[Flickr]({metadata['source_url']}) | [{metadata['license_name']}]({metadata['license_url']}) | "
        "Model prediction/chart overlay in generated PNG evidence |"
    )
(OUTPUT_DIR / "IMAGE_ATTRIBUTION.md").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")
images = {sample_id: Image.open(IMAGE_DIR / f"{sample_id}.jpg").convert("RGB") for sample_id in SAMPLES}

# Classification
print("\n--- Running Classification ---")
classification_model = YOLO(CLASSIFICATION_MODEL_FILE)
classification_model_sha256 = MODEL_SHA256[CLASSIFICATION_MODEL_FILE]
classification_records = []
for sample_id, image in images.items():
    result = classification_model.predict(source=image, device=DEVICE, verbose=False)[0]
    probs = result.probs
    top_ids = probs.top5
    top_scores = probs.top5conf.tolist()
    names = result.names
    for rank, (class_id, score) in enumerate(zip(top_ids, top_scores), start=1):
        classification_records.append({
            "sample_id": sample_id, "coco_image_id": SAMPLES[sample_id]["coco_image_id"],
            "image_width": image.width, "image_height": image.height,
            "task": "image_classification", "taxonomy_name": "ImageNet-1K",
            "model_file": CLASSIFICATION_MODEL_FILE, "model_sha256": classification_model_sha256,
            "ultralytics_version": ultralytics.__version__, "rank": rank,
            "class_id": int(class_id), "class_name": names[int(class_id)],
            "score": round(float(score), 6),
        })
with open(OUTPUT_DIR / "classification_predictions.json", "w", encoding="utf-8") as file:
    json.dump(classification_records, file, indent=2, ensure_ascii=False)

sample_id = "traffic"
rows = [row for row in classification_records if row["sample_id"] == sample_id]
fig, (image_ax, chart_ax) = plt.subplots(1, 2, figsize=(12, 5))
image_ax.imshow(images[sample_id])
image_ax.axis("off")
image_ax.set_title(f"Phân loại bằng {CLASSIFICATION_MODEL_FILE}: {sample_id}")
chart_ax.barh([r["class_name"] for r in rows][::-1], [r["score"] for r in rows][::-1], color="#e45756")
chart_ax.set_xlim(0, 1)
chart_ax.set_xlabel("độ tin cậy của mô hình (không phải chất lượng ground truth)")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "classification_top5.png", dpi=160, bbox_inches="tight")
plt.close()
del classification_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Detection
print("\n--- Running Detection ---")
detection_model = YOLO(DETECTION_MODEL_FILE)
detection_model_sha256 = MODEL_SHA256[DETECTION_MODEL_FILE]
DETECTION_SCORE_THRESHOLD = 0.35

def run_detection(sample_id, image, threshold):
    result = detection_model.predict(source=image, conf=threshold, device=DEVICE, verbose=False)[0]
    records = []
    if result.boxes is None: return records
    for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        x_min_raw, y_min_raw, x_max_raw, y_max_raw = [float(v) for v in box]
        x_min, y_min, x_max, y_max = [round(v, 2) for v in (x_min_raw, y_min_raw, x_max_raw, y_max_raw)]
        class_id = int(cls)
        records.append({
            "sample_id": sample_id, "coco_image_id": SAMPLES[sample_id]["coco_image_id"],
            "image_width": image.width, "image_height": image.height,
            "task": "object_detection", "taxonomy_name": "COCO-80",
            "model_file": DETECTION_MODEL_FILE, "model_sha256": detection_model_sha256,
            "ultralytics_version": ultralytics.__version__, "score_threshold": threshold,
            "class_id": class_id, "class_name": result.names[class_id], "score": round(float(conf), 6),
            "coordinate_unit": "pixel", "bbox_format": "xyxy",
            "bbox_xyxy": [x_min, y_min, x_max, y_max],
            "bbox_width": round(x_max_raw-x_min_raw, 2), "bbox_height": round(y_max_raw-y_min_raw, 2)
        })
    return records

detection_records = []
for sample_id, image in images.items():
    detection_records.extend(run_detection(sample_id, image, DETECTION_SCORE_THRESHOLD))
with open(OUTPUT_DIR / "detection_predictions.json", "w", encoding="utf-8") as file:
    json.dump(detection_records, file, indent=2, ensure_ascii=False)

sample_id = "kitchen"
result = detection_model.predict(source=images[sample_id], conf=DETECTION_SCORE_THRESHOLD, device=DEVICE, verbose=False)[0]
annotated = result.plot()[:, :, ::-1]
plt.figure(figsize=(12, 8))
plt.imshow(annotated)
plt.axis("off")
plt.title(f"Phát hiện vật thể bằng {DETECTION_MODEL_FILE}")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "detection_predictions.png", dpi=160, bbox_inches="tight")
plt.close()

for threshold in (0.20, 0.35, 0.60):
    rows_thresh = run_detection(sample_id, images[sample_id], threshold)
    print(f"threshold={threshold:.2f}: {len(rows_thresh)} vật thể → {[row['class_name'] for row in rows_thresh]}")

del detection_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Segmentation
print("\n--- Running Segmentation ---")
segmentation_model = YOLO(SEGMENTATION_MODEL_FILE)
segmentation_model_sha256 = MODEL_SHA256[SEGMENTATION_MODEL_FILE]
SEGMENTATION_SCORE_THRESHOLD = 0.35
segmentation_records = []
for sample_id, image in images.items():
    result = segmentation_model.predict(source=image, conf=SEGMENTATION_SCORE_THRESHOLD, device=DEVICE, verbose=False)[0]
    if result.boxes is None or result.masks is None: continue
    polygons = result.masks.xy
    for index, (box, cls, conf, polygon) in enumerate(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), polygons)):
        x_min, y_min, x_max, y_max = [round(float(v), 2) for v in box]
        class_id = int(cls)
        polygon_xy = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]
        segmentation_records.append({
            "sample_id": sample_id, "coco_image_id": SAMPLES[sample_id]["coco_image_id"],
            "image_width": image.width, "image_height": image.height,
            "task": "instance_segmentation", "taxonomy_name": "COCO-80",
            "model_file": SEGMENTATION_MODEL_FILE, "model_sha256": segmentation_model_sha256,
            "ultralytics_version": ultralytics.__version__, "score_threshold": SEGMENTATION_SCORE_THRESHOLD,
            "instance_id": f"{sample_id}-{index + 1:03d}",
            "class_id": class_id, "class_name": result.names[class_id], "score": round(float(conf), 6),
            "coordinate_unit": "pixel", "bbox_format": "xyxy",
            "bbox_xyxy": [x_min, y_min, x_max, y_max],
            "polygon_point_count": len(polygon_xy), "polygon_xy": polygon_xy
        })
with open(OUTPUT_DIR / "segmentation_predictions.json", "w", encoding="utf-8") as file:
    json.dump(segmentation_records, file, indent=2, ensure_ascii=False)

sample_id = "kitchen"
result = segmentation_model.predict(source=images[sample_id], conf=SEGMENTATION_SCORE_THRESHOLD, device=DEVICE, verbose=False)[0]
annotated = result.plot()[:, :, ::-1]
plt.figure(figsize=(12, 8))
plt.imshow(annotated)
plt.axis("off")
plt.title(f"Phân đoạn đối tượng bằng {SEGMENTATION_MODEL_FILE}")
plt.tight_layout()
plt.savefig(VISUAL_DIR / "segmentation_prediction.png", dpi=160, bbox_inches="tight")
plt.close()

del segmentation_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Validation
print("\n--- Running Validation ---")
provenance_fields = {"sample_id", "coco_image_id", "image_width", "image_height", "task", "taxonomy_name", "model_file", "model_sha256", "ultralytics_version"}
required_json = {
    "classification_predictions.json": provenance_fields | {"class_id", "class_name", "rank", "score"},
    "detection_predictions.json": provenance_fields | {"class_id", "class_name", "score", "score_threshold", "coordinate_unit", "bbox_format", "bbox_xyxy", "bbox_width", "bbox_height"},
    "segmentation_predictions.json": provenance_fields | {"instance_id", "class_id", "class_name", "score", "score_threshold", "coordinate_unit", "bbox_format", "bbox_xyxy", "polygon_point_count", "polygon_xy"},
}
expected_model_by_file = {
    "classification_predictions.json": CLASSIFICATION_MODEL_FILE,
    "detection_predictions.json": DETECTION_MODEL_FILE,
    "segmentation_predictions.json": SEGMENTATION_MODEL_FILE,
}
expected_samples = set(SAMPLES)
attribution_path = OUTPUT_DIR / "IMAGE_ATTRIBUTION.md"
if not attribution_path.is_file(): raise AssertionError("Thiếu IMAGE_ATTRIBUTION.md")
attribution_text = attribution_path.read_text(encoding="utf-8")
for metadata in SAMPLES.values():
    for required_attribution in (metadata["source_creator"], metadata["source_url"], metadata["license_url"]):
        if required_attribution not in attribution_text:
            raise AssertionError(f"Attribution thiếu: {required_attribution}")

required_png = [
    "visuals/classification_top5.png",
    "visuals/detection_predictions.png",
    "visuals/segmentation_prediction.png",
]
validated_rows = {}
for filename, expected_fields in required_json.items():
    path = OUTPUT_DIR / filename
    if not path.is_file(): raise AssertionError(f"Thiếu {path}")
    with open(path, encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or not rows: raise AssertionError(f"{filename} phải là list không rỗng")
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict): raise AssertionError(f"{filename} row {row_number} không phải object")
        missing_fields = expected_fields - set(row)
        if missing_fields: raise AssertionError(f"{filename} row {row_number} thiếu: {sorted(missing_fields)}")
        sample_id = row["sample_id"]
        if sample_id not in SAMPLES: raise AssertionError(f"{filename} có sample lạ: {sample_id}")
        if row.get("coco_image_id") != SAMPLES[sample_id]["coco_image_id"]:
            raise AssertionError(f"{filename} row {row_number} sai coco_image_id")
        expected_model = expected_model_by_file[filename]
        if row["model_file"] != expected_model or row.get("model_sha256") != MODEL_SHA256[expected_model]:
            raise AssertionError(f"{filename} row {row_number} sai model provenance")
        if not 0.0 <= float(row["score"]) <= 1.0:
            raise AssertionError(f"{filename} row {row_number} có score ngoài [0, 1]")
    actual_samples = {row["sample_id"] for row in rows}
    if actual_samples != expected_samples:
        raise AssertionError(f"{filename} phải có đủ sample: {sorted(expected_samples)}")
    validated_rows[filename] = rows

classification_rows = validated_rows["classification_predictions.json"]
for sample_id in expected_samples:
    ranks = sorted(row["rank"] for row in classification_rows if row["sample_id"] == sample_id)
    if ranks != [1, 2, 3, 4, 5]: raise AssertionError(f"Top-5 không hợp lệ cho {sample_id}: {ranks}")

for filename in ("detection_predictions.json", "segmentation_predictions.json"):
    for row_number, row in enumerate(validated_rows[filename], start=1):
        width, height = row["image_width"], row["image_height"]
        bbox = row["bbox_xyxy"]
        if len(bbox) != 4: raise AssertionError(f"{filename} row {row_number} bbox phải có 4 số")
        x_min, y_min, x_max, y_max = bbox
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise AssertionError(f"{filename} row {row_number} bbox ngoài ảnh hoặc không dương")

segmentation_rows = validated_rows["segmentation_predictions.json"]
for row_number, row in enumerate(segmentation_rows, start=1):
    polygon = row["polygon_xy"]
    if len(polygon) < 3 or row.get("polygon_point_count") != len(polygon):
        raise AssertionError(f"Segmentation row {row_number} có polygon không hợp lệ")
    for point in polygon:
        if len(point) != 2 or not (0 <= point[0] <= row["image_width"] and 0 <= point[1] <= row["image_height"]):
            raise AssertionError(f"Segmentation row {row_number} có điểm polygon ngoài ảnh")

for filename in required_png:
    path = OUTPUT_DIR / filename
    if not path.is_file() or path.stat().st_size < 1000: raise AssertionError(f"Thiếu hoặc lỗi {path}")
    with Image.open(path) as image:
        image.load()
        if image.width < 100 or image.height < 100: raise AssertionError(f"Ảnh quá nhỏ: {path}")

instance_ids = [row["instance_id"] for row in segmentation_rows]
if len(instance_ids) != len(set(instance_ids)): raise AssertionError("instance_id phải duy nhất trong output")

print(f"PASS: đủ {len(required_json)} JSON, {len(required_png)} PNG, attribution và instance_id duy nhất.")
print("Hoàn thành tạo evidence thành công!")
