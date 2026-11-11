"""
Vehicle + number-plate pipeline:
    vehicle detector (YOLOv8n)  ->  car / motorcycle / bus / truck / bicycle  + PCE
    plate detector   (YOLOv8n)  ->  plate boxes
    OCR              (EasyOCR)  ->  plate text
Used by both main.py (the API) and test_on_images.py (batch testing).
"""
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", str(BASE / "models" / "vehicle_best.pt"))
PLATE_MODEL = os.getenv("PLATE_MODEL", str(BASE / "models" / "plate_best.pt"))
VEHICLE_CONF = float(os.getenv("VEHICLE_CONF", "0.35"))
PLATE_CONF = float(os.getenv("PLATE_CONF", "0.30"))

# Passenger Car Equivalent (PCE) factors.
# PLACEHOLDER VALUES - replace them with the factors your project / IRC reference specifies.
# PLACEHOLDER values only. Confirm/replace using the PCE standard selected for your campus study.
PCE_FACTORS = {"car": 1.0, "motorcycle": 0.5, "bus": 3.0, "truck": 3.0, "bicycle": 0.5}

OCR_ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Typical Indian formats, e.g. HP33A1234 / MH12AB1234 and the BH series 22BH1234AA (informational only)
PLATE_PATTERNS = [re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"), re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")]

COLORS = {"car": (31, 119, 180), "motorcycle": (255, 127, 14), "bus": (44, 160, 44), "truck": (214, 39, 40), "bicycle": (148, 103, 189)}


def load_image(source) -> Image.Image:
    """Open an image (path or file object), fix phone rotation (EXIF) and return RGB."""
    img = Image.open(source)
    img.load()
    return ImageOps.exif_transpose(img).convert("RGB")


def get_font(size, bold=False):
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


class ANPRPipeline:
    def __init__(self, vehicle_path=VEHICLE_MODEL, plate_path=PLATE_MODEL, use_ocr=True):
        import torch
        from ultralytics import YOLO

        self.use_gpu = torch.cuda.is_available()
        self.device = 0 if self.use_gpu else "cpu"

        self.vehicle = YOLO(vehicle_path) if Path(vehicle_path).exists() else None
        self.plate = YOLO(plate_path) if Path(plate_path).exists() else None
        if self.vehicle is None:
            print(f"WARNING: vehicle model not found at {vehicle_path}")
        if self.plate is None:
            print(f"WARNING: plate model not found at {plate_path}")

        self.reader = None
        if use_ocr and self.plate is not None:
            try:
                import easyocr  # downloads its language model the first time (needs internet once)
                self.reader = easyocr.Reader(["en"], gpu=self.use_gpu)
            except Exception as e:
                print(f"[ANPR Warning] EasyOCR model not downloaded yet ({e}). OCR reading disabled, vehicle and plate detection active.")
                self.reader = None

    # ------------------------------------------------------------------ detection
    @staticmethod
    def _boxes(result, keep=None):
        out = []
        for b in result.boxes:
            cls = result.names[int(b.cls[0])]
            if keep is not None and not keep(cls):
                continue
            x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            out.append({"class": cls, "confidence": round(float(b.conf[0]), 4), "box": [x1, y1, x2, y2]})
        return out

    def detect_vehicles(self, img, conf=None):
        res = self.vehicle.predict(img, conf=conf or VEHICLE_CONF, device=self.device, verbose=False)[0]
        return self._boxes(res)

    def detect_plates(self, img, conf=None):
        res = self.plate.predict(img, conf=conf or PLATE_CONF, device=self.device, verbose=False)[0]
        has_plate_class = any("plate" in str(n).lower() for n in res.names.values())
        keep = (lambda c: "plate" in c.lower()) if has_plate_class else None  # ignore a 'Vehicle' class if present
        return self._boxes(res, keep)

    # ------------------------------------------------------------------ OCR
    def read_plate(self, crop):
        """Read the text on a cropped plate. Handles one-line and two-line plates."""
        empty = {"text": "", "ocr_confidence": 0.0, "format_valid": False}
        if self.reader is None:
            return empty
        import numpy as np

        w, h = crop.size
        if h < 80:  # small crops read badly: enlarge
            crop = crop.resize((max(int(w * 80 / h), 1), 80))
        results = self.reader.readtext(np.array(crop.convert("L")), allowlist=OCR_ALLOWED, detail=1)
        if not results:
            return empty

        items = []  # (y_center, x_left, text, confidence, height)
        for box, text, conf in results:
            ys, xs = [p[1] for p in box], [p[0] for p in box]
            items.append((sum(ys) / len(ys), min(xs), text, float(conf), max(ys) - min(ys)))
        items.sort(key=lambda t: t[0])
        med_h = sorted(t[4] for t in items)[len(items) // 2] or 1
        lines, cur = [], [items[0]]
        for it in items[1:]:  # group pieces into text lines (two-line plates)
            if abs(it[0] - cur[-1][0]) > 0.6 * med_h:
                lines.append(cur)
                cur = [it]
            else:
                cur.append(it)
        lines.append(cur)
        text = "".join("".join(t[2] for t in sorted(line, key=lambda t: t[1])) for line in lines)
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        return {
            "text": text,
            "ocr_confidence": round(sum(t[3] for t in items) / len(items), 4),
            "format_valid": any(p.match(text) for p in PLATE_PATTERNS),
        }

    def plates_with_text(self, img, conf=None, read_text=True):
        plates = self.detect_plates(img, conf)
        for p in plates:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            px, py = int(0.03 * (x2 - x1)), int(0.08 * (y2 - y1))
            crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(img.width, x2 + px), min(img.height, y2 + py)))
            p["crop"] = crop  # kept for saving; removed by public()
            p.update(self.read_plate(crop) if read_text else {"text": "", "ocr_confidence": 0.0, "format_valid": False})
        return plates

    # ------------------------------------------------------------------ full analysis
    def analyze(self, img, vehicle_conf=None, plate_conf=None, read_text=True):
        vehicles = self.detect_vehicles(img, vehicle_conf) if self.vehicle else []
        plates = self.plates_with_text(img, plate_conf, read_text) if self.plate else []

        for v in vehicles:
            v["pce"] = PCE_FACTORS.get(v["class"], 1.0)
            v["plate"] = None
        unassigned = []
        for p in plates:  # a plate belongs to the smallest vehicle box that contains its centre
            cx, cy = (p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2
            best, best_area = None, None
            for v in vehicles:
                x1, y1, x2, y2 = v["box"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best_area:
                        best, best_area = v, area
            if best is None:
                unassigned.append(p)
            elif best["plate"] is None or p["confidence"] > best["plate"]["confidence"]:
                best["plate"] = p

        counts = {}
        for v in vehicles:
            counts[v["class"]] = counts.get(v["class"], 0) + 1
        return {
            "image_size": [img.width, img.height],
            "vehicle_count": len(vehicles),
            "counts": counts,
            "pce_total": round(sum(v["pce"] for v in vehicles), 2),
            "vehicles": vehicles,
            "unassigned_plates": unassigned,
        }

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def draw(img, result):
        img = img.copy()
        d = ImageDraw.Draw(img)
        font = get_font(max(14, img.width // 70), bold=True)

        def label(x, y, text, color):
            tb = d.textbbox((x, y), text, font=font)
            d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
            d.text((x, y), text, font=font, fill="white")

        for v in result["vehicles"]:
            color = COLORS.get(v["class"], (120, 120, 120))
            x1, y1, x2, y2 = v["box"]
            d.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 400))
            label(x1 + 3, max(y1 - font.size - 6, 0), f"{v['class']} {v['confidence'] * 100:.0f}%", color)
        plates = [v["plate"] for v in result["vehicles"] if v["plate"]] + result["unassigned_plates"]
        for p in plates:
            x1, y1, x2, y2 = p["box"]
            d.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=max(2, img.width // 400))
            label(x1, min(y2 + 4, img.height - font.size - 6), p.get("text") or "plate", (170, 120, 0))
        return img


def public(result):
    """Copy of an analysis result without PIL crops, safe to send as JSON."""
    def clean_plate(p):
        return None if p is None else {k: v for k, v in p.items() if k != "crop"}
    out = dict(result)
    out["vehicles"] = [{**v, "plate": clean_plate(v["plate"])} for v in result["vehicles"]]
    out["unassigned_plates"] = [clean_plate(p) for p in result["unassigned_plates"]]
    return out
