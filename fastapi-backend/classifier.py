"""Loads best_model.pth (EfficientNetV2-S) and classifies road hazards.
Directly incorporates RD_API PyTorch architecture with enriched metadata.
"""
import os
import io
import json
import base64
import urllib.request
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, UnidentifiedImageError

# PyTorch & Torchvision
HAVE_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAVE_TORCH = True
except Exception as e:
    HAVE_TORCH = False

CLASSES_DEFAULT = [
    "Broken_Staircase",
    "Debris_on_road",
    "Edge_Shoulder_Damage",
    "Iron_grid_damage",
    "Longitudinal_Crack",
    "Map(Block)_Cracking",
    "Patch_Repair",
    "Paver_Damage",
    "Pothole_Surface_Breakout",
    "Raveling_Loose_Aggregate",
    "Rubble_Unpaved_surface",
    "Transverse_Crack",
    "Vegetation_Moss",
]

LABEL_FORMATTING = {
    "Broken_Staircase": "Broken Staircase / Curbside",
    "Debris_on_road": "Debris on Road",
    "Edge_Shoulder_Damage": "Edge & Shoulder Damage",
    "Iron_grid_damage": "Drainage Iron Grid Damage",
    "Longitudinal_Crack": "Longitudinal Road Crack",
    "Map(Block)_Cracking": "Map / Block Cracking",
    "Patch_Repair": "Patch Repair Deterioration",
    "Paver_Damage": "Paver / Brick Road Damage",
    "Pothole_Surface_Breakout": "Pothole & Surface Breakout",
    "Raveling_Loose_Aggregate": "Raveling & Loose Aggregate",
    "Rubble_Unpaved_surface": "Rubble & Unpaved Surface",
    "Transverse_Crack": "Transverse Road Crack",
    "Vegetation_Moss": "Vegetation & Moss Overgrowth",
}

SEVERITY_MAPPING = {
    "Pothole_Surface_Breakout": "critical",
    "Broken_Staircase": "high",
    "Iron_grid_damage": "high",
    "Map(Block)_Cracking": "high",
    "Longitudinal_Crack": "medium",
    "Transverse_Crack": "medium",
    "Paver_Damage": "medium",
    "Edge_Shoulder_Damage": "medium",
    "Debris_on_road": "medium",
    "Rubble_Unpaved_surface": "medium",
    "Patch_Repair": "low",
    "Raveling_Loose_Aggregate": "low",
    "Vegetation_Moss": "low",
}

CATEGORY_MAPPING = {
    "Broken_Staircase": "Broken_Staircase",
    "Debris_on_road": "Debris_on_road",
    "Edge_Shoulder_Damage": "Edge_Shoulder_Damage",
    "Iron_grid_damage": "Iron_grid_damage",
    "Longitudinal_Crack": "Longitudinal_Crack",
    "Map(Block)_Cracking": "Map(Block)_Cracking",
    "Patch_Repair": "Patch_Repair",
    "Paver_Damage": "Paver_Damage",
    "Pothole_Surface_Breakout": "Pothole_Surface_Breakout",
    "Raveling_Loose_Aggregate": "Raveling_Loose_Aggregate",
    "Rubble_Unpaved_surface": "Rubble_Unpaved_surface",
    "Transverse_Crack": "Transverse_Crack",
    "Vegetation_Moss": "Vegetation_Moss",
}

LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE", "0.60"))


class Classifier:
    """Core PyTorch EfficientNetV2-S classifier from RD_API."""
    def __init__(self, model_path: str):
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch is not available in current python environment.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(model_path, map_location=self.device)
        self.classes = ck.get("classes", CLASSES_DEFAULT)
        self.img_size = ck.get("img_size", 300)

        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(self.classes))
        )
        model.load_state_dict(ck["model"])
        self.model = model.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, img: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        """img: PIL.Image -> list of (class_name, probability), highest first."""
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        conf, idx = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], float(c)) for c, i in zip(conf, idx)]


class RoadHazardClassifier:
    """Singleton wrapper providing unified prediction API matching both RD_API and RoadSentinel formats."""
    _instance = None

    @classmethod
    def get_instance(cls, model_path: str = None) -> "RoadHazardClassifier":
        if cls._instance is None:
            if model_path is None:
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                candidates = [
                    os.path.join(curr_dir, "best_model.pth"),
                    os.path.join(curr_dir, "..", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "RD_API", "RD_API", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "DP-PROJECT", "fastapi-backend", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "Backend", "best_model.pth"),
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        model_path = c
                        break
            cls._instance = RoadHazardClassifier(model_path)
        return cls._instance

    def __init__(self, model_path: str = None):
        self.classes = CLASSES_DEFAULT
        self.img_size = 300
        self.clf = None

        if model_path and os.path.isfile(model_path) and HAVE_TORCH:
            try:
                print(f"[ML Model] Loading EfficientNetV2-S from {model_path}...")
                self.clf = Classifier(model_path)
                self.classes = self.clf.classes
                self.img_size = self.clf.img_size
                print(f"[ML Model] Model ready on device: {self.clf.device} with {len(self.classes)} classes.")
            except Exception as e:
                print(f"[ML Model Warning] PyTorch model initialization failed: {e}")
                self.clf = None

    @property
    def device(self):
        return self.clf.device if self.clf else "cpu"

    def _load_image(self, img_input: Any) -> Image.Image:
        """Converts diverse image formats (PIL, bytes, base64, URLs, disk paths) into PIL.Image."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_input)).convert("RGB")

        if isinstance(img_input, str):
            clean = img_input.strip()

            # Base64 data URI
            if clean.startswith("data:image"):
                if "base64," in clean:
                    base64_data = clean.split("base64,", 1)[1]
                    img_bytes = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # HTTP URL
            if clean.startswith("http://") or clean.startswith("https://"):
                try:
                    req = urllib.request.Request(clean, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        return Image.open(io.BytesIO(resp.read())).convert("RGB")
                except Exception as e:
                    print(f"[ML Model Warning] Could not download image URL: {e}")

            # Direct base64 string
            if len(clean) > 100 and not os.path.exists(clean):
                try:
                    img_bytes = base64.b64decode(clean)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    pass

            # Direct disk file path
            if os.path.isfile(clean):
                return Image.open(clean).convert("RGB")

            # Relative upload search
            rel = clean.lstrip("/\\")
            base_search_dirs = [
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "Backend"),
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "DP-PROJECT", "backend"),
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "DP-PROJECT", "fastapi-backend"),
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "RoadSentinel", "Backend"),
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),
            ]
            for b in base_search_dirs:
                direct = os.path.join(b, rel)
                if os.path.isfile(direct):
                    return Image.open(direct).convert("RGB")
                in_uploads = os.path.join(b, "uploads", os.path.basename(rel))
                if os.path.isfile(in_uploads):
                    return Image.open(in_uploads).convert("RGB")

        raise ValueError("Could not decode image input into a valid image.")

    def predict(self, img_input: Any, top_k: int = 3, filename: str = "road.jpg") -> Dict[str, Any]:
        """
        Executes inference and returns exact response structure requested by user:
        {
          "filename": "road.jpg",
          "prediction": "Pothole_Surface_Breakout",
          "confidence": 0.9123,
          "low_confidence": false,
          "top_predictions": [ ... ]
        }
        with additional backwards-compatible fields (success, rawClass, label, etc.).
        """
        try:
            pil_img = self._load_image(img_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid image format: {str(e)}",
                "filename": filename,
                "prediction": "other",
                "confidence": 0.0,
                "low_confidence": True,
                "top_predictions": [],
            }

        if self.clf is not None:
            try:
                top = self.clf.predict(pil_img, top_k=top_k)
            except Exception as e:
                print(f"[ML Model Error] PyTorch inference execution failed: {e}")
                top = [(self.classes[0], 0.08)]
        else:
            top = [(self.classes[0], 0.08)]

        top_label, top_conf = top[0]
        is_low = top_conf < LOW_CONFIDENCE_THRESHOLD

        top_predictions_list = []
        for c, p in top:
            top_predictions_list.append({
                "class": c,
                "confidence": round(float(p), 4),
                "rawClass": c,
                "label": LABEL_FORMATTING.get(c, c.replace("_", " ")),
                "confidencePercentage": f"{round(float(p) * 100, 1)}%",
                "mappedCategory": CATEGORY_MAPPING.get(c, "other"),
                "suggestedSeverity": SEVERITY_MAPPING.get(c, "medium"),
            })

        best = top_predictions_list[0]
        return {
            "filename": filename,
            "prediction": top_label,
            "confidence": round(float(top_conf), 4),
            "low_confidence": is_low,
            "top_predictions": [
                {"class": item["class"], "confidence": item["confidence"]}
                for item in top_predictions_list
            ],
            # Enriched fields for existing UI components
            "success": True,
            "rawClass": top_label,
            "label": best["label"],
            "confidencePercentage": best["confidencePercentage"],
            "mappedCategory": best["mappedCategory"],
            "suggestedSeverity": best["suggestedSeverity"],
            "topPredictions": top_predictions_list,
            "isLowConfidence": is_low,
        }


if __name__ == "__main__":
    import sys
    classifier = RoadHazardClassifier.get_instance()

    if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
        input_data = sys.stdin.read().strip()
        res = classifier.predict(input_data)
        print(json.dumps(res, indent=2))
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        res = classifier.predict(sys.argv[1])
        print(json.dumps(res, indent=2))
    else:
        dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        res = classifier.predict(dummy_img)
        print("Model verification test result:")
        print(json.dumps(res, indent=2))
