import os
import re
import base64
import uuid
import json
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
from app.models import SpeedBreachPayload, ProcessedBreachRecord, ActiveLearningSample

# Directories
EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "evidence")
ACTIVE_LEARNING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "active_learning")

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(ACTIVE_LEARNING_DIR, exist_ok=True)

# Passenger Car Equivalent (PCE) Weightings
PCE_WEIGHTS = {
    "Car": 1.0,
    "SUV": 1.2,
    "Motorcycle": 0.5,
    "Bus": 3.0,
    "Truck": 3.0,
    "Van": 1.5,
}

# Simulated Indian / Himachal Pradesh plate formats for campus validation
SAMPLE_PLATES = [
    "HP 33 B 4921",
    "HP 65 A 1084",
    "DL 8C AK 9012",
    "CH 01 BG 7733",
    "HP 28 T 3319",
    "HR 26 DQ 5521",
    "HP 31 E 8840",
]

def save_base64_image(base64_str: str, device_id: str, breach_id: str) -> Optional[str]:
    """
    Decodes 1080p JPEG from RAM buffer directly to structured NVMe storage.
    Path: storage/evidence/{device_id}/{YYYY-MM-DD}/{breach_id}.jpg
    """
    try:
        # Strip header if present e.g. "data:image/jpeg;base64,"
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_str)
        date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = os.path.join(EVIDENCE_DIR, device_id, date_folder)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{breach_id}.jpg"
        file_path = os.path.join(target_dir, filename)

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # Return relative URL for web viewing
        return f"/evidence/{device_id}/{date_folder}/{filename}"
    except Exception as e:
        print(f"[Evidence Storage Error] {e}")
        return None

def run_ml_inference(payload: SpeedBreachPayload, image_path: Optional[str]) -> Tuple[str, float, str, float]:
    """
    Simulates / Executes YOLOv8 Vehicle Classification + ALPR (License Plate OCR) pipeline.
    Returns: (vehicle_type, pce_weight, license_plate, ocr_confidence)
    """
    import random

    # Deterministic or randomized realistic classification based on speed profile
    if payload.speed_kmh > 65.0:
        vehicle_type = random.choice(["Car", "SUV", "Motorcycle"])
    elif payload.speed_kmh < 25.0:
        vehicle_type = random.choice(["Truck", "Bus", "Van"])
    else:
        vehicle_type = random.choice(["Car", "SUV", "Motorcycle", "Van"])

    pce = PCE_WEIGHTS.get(vehicle_type, 1.0)

    # Realistic OCR detection: High confidence for most, occasional blurry plate for active learning
    confidence = round(random.uniform(0.62, 0.98), 2)
    plate = random.choice(SAMPLE_PLATES)

    return vehicle_type, pce, plate, confidence

def process_speed_breach(payload: SpeedBreachPayload) -> Tuple[ProcessedBreachRecord, Optional[ActiveLearningSample]]:
    """
    End-to-end speed breach processing:
    1. Generates unique breach identifier
    2. Decodes and archives 1080p snapshot
    3. Runs vehicle classification & license plate OCR
    4. Computes delta over posted limit & severity rating
    5. Flags low confidence predictions (< 75%) to Active Learning Queue
    """
    breach_id = f"BREACH-{uuid.uuid4().hex[:8].upper()}"
    timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()
    posted_limit = payload.posted_limit_kmh or 35.0
    delta = round(payload.speed_kmh - posted_limit, 1)

    # Calculate severity rating
    if delta >= 30.0:
        severity = "critical"
    elif delta >= 15.0:
        severity = "high"
    else:
        severity = "moderate"

    # Save evidence image
    image_url = None
    if payload.image_base64:
        image_url = save_base64_image(payload.image_base64, payload.device_id, breach_id)

    # ML Inference
    vehicle_type, pce_weight, plate, ocr_conf = run_ml_inference(payload, image_url)

    # Active Learning Pipeline check: Confidence < 75% triggers retraining queue
    routed_to_al = False
    al_sample: Optional[ActiveLearningSample] = None

    if ocr_conf < 0.75:
        routed_to_al = True
        al_sample = ActiveLearningSample(
            sample_id=f"AL-{uuid.uuid4().hex[:6].upper()}",
            timestamp=timestamp,
            device_id=payload.device_id,
            image_path=image_url or "no_image",
            predicted_plate=plate,
            confidence=ocr_conf,
            vehicle_type=vehicle_type,
            flag_reason=f"OCR Confidence ({int(ocr_conf*100)}%) fell below 75% enterprise threshold.",
        )
        # Store in active learning index file
        al_log_file = os.path.join(ACTIVE_LEARNING_DIR, "queue.jsonl")
        try:
            with open(al_log_file, "a") as f:
                f.write(json.dumps(al_sample.model_dump()) + "\n")
        except Exception as e:
            print(f"[Active Learning Queue Write Error] {e}")

    record = ProcessedBreachRecord(
        breach_id=breach_id,
        device_id=payload.device_id,
        timestamp=timestamp,
        speed_kmh=payload.speed_kmh,
        posted_limit_kmh=posted_limit,
        delta_over_limit=delta,
        severity=severity,
        vehicle_type=vehicle_type,
        pce_weight=pce_weight,
        license_plate=plate,
        ocr_confidence=ocr_conf,
        routed_to_active_learning=routed_to_al,
        image_url=image_url,
        fine_suppressed=False,
        road_name=payload.road_name or "Campus Road",
        latitude=payload.latitude or 31.7807,
        longitude=payload.longitude or 76.9967,
    )

    return record, al_sample
