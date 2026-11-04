"""
Raspberry Pi 4 Edge Sensor Simulator (HLK-LD2415H 24GHz Doppler Radar + Camera Module 3)
Streams live zero-AI radar state machine telemetry and speed breach alerts to FastAPI.
"""

import time
import json
import random
import base64
import requests
from datetime import datetime, timezone, timedelta
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

FASTAPI_URL = "http://localhost:8000/api/v1/telemetry"

NODES = [
    {
        "device_id": "RPI4_NORTH_GATE",
        "road_name": "North Campus Main Gate Road",
        "posted_limit": 30.0,
        "lat": 31.7816,
        "lng": 76.9972,
        "base_speed": 28.0,
        "traffic_rate": 6,  # vehicles per 10s cycle
    },
    {
        "device_id": "RPI4_SOUTH_ACADEMIC",
        "road_name": "South Campus Academic Complex Road",
        "posted_limit": 35.0,
        "lat": 31.7754,
        "lng": 76.9861,
        "base_speed": 34.0,
        "traffic_rate": 8,
    }
]

def generate_mock_vehicle_snapshot(plate_text: str, speed: float, vehicle_type: str) -> str:
    """Generates a synthetic camera snapshot encoded as Base64 JPEG."""
    img = Image.new("RGB", (640, 360), color=(30, 35, 45))
    draw = ImageDraw.Draw(img)

    # Road background representation
    draw.rectangle([0, 180, 640, 360], fill=(50, 55, 65))
    draw.line([320, 180, 320, 360], fill=(240, 200, 50), width=4)

    # Vehicle body
    draw.rectangle([220, 160, 420, 290], fill=(30, 90, 170), outline=(20, 60, 120), width=3)
    draw.rectangle([250, 120, 390, 180], fill=(70, 130, 200)) # windshield

    # License Plate
    draw.rectangle([270, 250, 370, 280], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((280, 258), plate_text, fill=(0, 0, 0))

    # Telemetry OSD watermark
    draw.text((20, 20), f"RADAR SPEED: {speed:.1f} km/h | {vehicle_type}", fill=(255, 80, 80))
    draw.text((20, 40), f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill=(220, 220, 220))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def run_simulation(interval_seconds: int = 15):
    print("=" * 65)
    print("🚗 RoadSentinel Raspberry Pi Edge Telemetry Streamer Active")
    print(f"📡 Target Server: {FASTAPI_URL}")
    print(f"⏱️ Periodic Telemetry Interval: {interval_seconds} seconds")
    print("=" * 65)

    window_start = datetime.now(timezone.utc) - timedelta(seconds=interval_seconds)

    while True:
        try:
            window_end = datetime.now(timezone.utc)

            for node in NODES:
                # 1. Simulate Radar State Machine Tracking
                passes = random.randint(3, node["traffic_rate"])
                speeds = []
                approaching = 0
                receding = 0

                for _ in range(passes):
                    # Speed variation around base speed
                    spd = round(random.gauss(node["base_speed"], 6.5), 1)
                    spd = max(12.0, min(spd, 72.0))
                    speeds.append(spd)

                    if random.random() > 0.45:
                        approaching += 1
                    else:
                        receding += 1

                # 2. Package Periodic Summary Telemetry
                occupancy = round(min(85.0, (passes * 1.8 / interval_seconds) * 100.0), 1)
                summary_payload = {
                    "device_id": node["device_id"],
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "raw_vehicle_count": len(speeds),
                    "speed_array": speeds,
                    "approaching_count": approaching,
                    "receding_count": receding,
                    "radar_occupancy_pct": occupancy,
                    "posted_speed_limit": node["posted_limit"],
                    "road_name": node["road_name"],
                    "latitude": node["lat"],
                    "longitude": node["lng"],
                }

                res = requests.post(f"{FASTAPI_URL}/summary", json=summary_payload, timeout=6)
                if res.status_code == 200:
                    d = res.json().get("data", {})
                    print(f"[{node['device_id']}] ✅ Summary: {d.get('vehicle_count')} vehicles | V85={d.get('speed_85th_percentile')} km/h | Flow={d.get('flow_rate_vph')} VPH | {d.get('congestion_status')}")
                else:
                    print(f"[{node['device_id']}] ⚠️ Summary failed: {res.status_code}")

                # 3. Simulate Occasional Speed Breach with Camera Snapshot
                high_speeds = [s for s in speeds if s >= node["posted_limit"] + 8.0]
                if high_speeds and random.random() > 0.4:
                    breach_speed = max(high_speeds)
                    plate = random.choice(["HP 33 B 4921", "HP 65 A 1084", "DL 8C AK 9012", "CH 01 BG 7733", "HP 31 E 8840"])
                    v_type = "Car" if breach_speed < 45 else "SUV"
                    b64_img = generate_mock_vehicle_snapshot(plate, breach_speed, v_type)

                    breach_payload = {
                        "device_id": node["device_id"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "speed_kmh": breach_speed,
                        "posted_limit_kmh": node["posted_limit"],
                        "road_name": node["road_name"],
                        "latitude": node["lat"],
                        "longitude": node["lng"],
                        "image_base64": b64_img,
                    }

                    b_res = requests.post(f"{FASTAPI_URL}/breach", json=breach_payload, timeout=8)
                    if b_res.status_code == 200:
                        b_data = b_res.json().get("data", {})
                        print(f"🚨 [BREACH DETECTED] {node['device_id']}: {breach_speed} km/h (Limit: {node['posted_limit']}) | Plate: {b_data.get('license_plate')} ({int(b_data.get('ocr_confidence',0)*100)}% conf)")

            window_start = window_end
            time.sleep(interval_seconds)

        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for FastAPI server on {FASTAPI_URL} to become ready...")
            time.sleep(4)
        except KeyboardInterrupt:
            print("\nSimulation stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_simulation(interval_seconds=12)
