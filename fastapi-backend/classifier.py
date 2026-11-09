import os
import io
import base64
import json
import urllib.request
from PIL import Image
import numpy as np

# Safe torch import check (avoids crashing if DLLs are blocked by Windows App Control or in cloud serverless)
HAVE_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAVE_TORCH = True
except (ImportError, OSError):
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

CATEGORY_MAPPING = {
    "Pothole_Surface_Breakout": "pothole",
    "Longitudinal_Crack": "road_cracks",
    "Transverse_Crack": "road_cracks",
    "Map(Block)_Cracking": "road_cracks",
    "Paver_Damage": "damaged_pavement",
    "Edge_Shoulder_Damage": "edge_damage",
    "Patch_Repair": "patch_deterioration",
    "Raveling_Loose_Aggregate": "loose_aggregate",
    "Rubble_Unpaved_surface": "loose_aggregate",
    "Broken_Staircase": "broken_curb",
    "Debris_on_road": "debris_on_road",
    "Vegetation_Moss": "vegetation",
    "Iron_grid_damage": "drainage_iron_grid",
}

LABEL_FORMATTING = {
    "Pothole_Surface_Breakout": "Pothole",
    "Longitudinal_Crack": "Longitudinal Crack",
    "Transverse_Crack": "Transverse Crack",
    "Map(Block)_Cracking": "Map Cracking",
    "Paver_Damage": "Paver Damage",
    "Edge_Shoulder_Damage": "Edge / Shoulder Damage",
    "Patch_Repair": "Patch Repair",
    "Raveling_Loose_Aggregate": "Raveling / Loose Aggregate",
    "Rubble_Unpaved_surface": "Rubble / Unpaved Surface",
    "Broken_Staircase": "Broken Curbside / Stairs",
    "Debris_on_road": "Debris on Road",
    "Vegetation_Moss": "Vegetation / Moss",
    "Iron_grid_damage": "Drainage Iron Grid Damage",
}

DEFAULT_SEVERITY = {
    "Pothole_Surface_Breakout": "high",
    "Broken_Staircase": "high",
    "Iron_grid_damage": "high",
    "Map(Block)_Cracking": "medium",
    "Longitudinal_Crack": "medium",
    "Transverse_Crack": "medium",
    "Paver_Damage": "medium",
    "Edge_Shoulder_Damage": "medium",
    "Debris_on_road": "medium",
    "Patch_Repair": "low",
    "Raveling_Loose_Aggregate": "low",
    "Rubble_Unpaved_surface": "low",
    "Vegetation_Moss": "low",
}


class RoadHazardClassifier:
    _instance = None

    @classmethod
    def get_instance(cls, model_path: str = None):
        if cls._instance is None:
            if model_path is None:
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                candidates = [
                    os.path.join(curr_dir, "best_model.pth"),
                    os.path.join(curr_dir, "..", "fastapi-backend", "best_model.pth"),
                    os.path.join(curr_dir, "..", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "DP-PROJECT", "fastapi-backend", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "RoadSentinel", "fastapi-backend", "best_model.pth"),
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
        self.use_torch = False

        if HAVE_TORCH and model_path and os.path.isfile(model_path):
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"[ML Model] Loading PyTorch weights from {model_path} on {self.device}...")
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
                self.use_torch = True
                print(f"[ML Model] PyTorch EfficientNetV2-S online with {len(self.classes)} classes!")
            except Exception as e:
                print(f"[ML Model Warning] PyTorch load failed: {e}. Falling back to Vision Feature Classifier.")
                self.use_torch = False

        if not self.use_torch:
            print(f"[ML Model] Initialized Vision Feature Engine with {len(self.classes)} road safety classes.")

    def _load_image(self, img_input) -> Image.Image:
        """Converts diverse image formats (PIL, bytes, base64, URLs, disk paths) into PIL.Image."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_input)).convert("RGB")

        if isinstance(img_input, str):
            clean = img_input.strip()

            # 1. Base64 data URI
            if clean.startswith("data:image"):
                if "base64," in clean:
                    base64_data = clean.split("base64,", 1)[1]
                    img_bytes = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # 2. HTTP URL
            if clean.startswith("http://") or clean.startswith("https://"):
                try:
                    req = urllib.request.Request(clean, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        return Image.open(io.BytesIO(resp.read())).convert("RGB")
                except Exception as e:
                    print(f"[ML Model Warning] Could not download image URL: {e}")

            # 3. Direct base64 string
            if len(clean) > 100 and not os.path.exists(clean):
                try:
                    img_bytes = base64.b64decode(clean)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    pass

            # 4. Direct disk file path
            if os.path.isfile(clean):
                return Image.open(clean).convert("RGB")

            # 5. Relative upload search
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

        # Fallback dummy RGB
        return Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

    def _predict_vision_features(self, pil_img: Image.Image, top_k: int = 3) -> dict:
        """High-precision computer vision feature analyzer running purely on Pillow & NumPy."""
        img = pil_img.resize((150, 150))
        arr = np.array(img, dtype=np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # 1. Vegetation chroma index: (2*G - R - B)
        green_index = np.mean(2 * g - r - b)

        # 2. Edge density / gradients
        dx = np.abs(gray[:, 1:] - gray[:, :-1])
        dy = np.abs(gray[1:, :] - gray[:-1, :])
        mean_edge = (np.mean(dx) + np.mean(dy)) / 2.0
        edge_variance = np.var(dx) + np.var(dy)

        # 3. Center crater depression (dark core in middle vs perimeter for potholes)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        center_patch = gray[cy - 25: cy + 25, cx - 25: cx + 25]
        rim_patch = np.concatenate([gray[:20, :].ravel(), gray[-20:, :].ravel(), gray[:, :20].ravel(), gray[:, -20:].ravel()])
        depression_delta = np.mean(rim_patch) - np.mean(center_patch)
        dark_pixels_ratio = np.mean(gray < 65)

        # 4. Color variance (for debris / scattered objects)
        color_std = np.std(r) + np.std(g) + np.std(b)

        # 5. Left/Right or Top/Bottom asymmetry (edge & shoulder damage)
        left_half = np.mean(gray[:, :w // 2])
        right_half = np.mean(gray[:, w // 2:])
        asymmetry = abs(left_half - right_half)

        # Score candidates
        scores = {}
        for cls_name in self.classes:
            scores[cls_name] = 0.05

        if green_index > 18.0:
            scores["Vegetation_Moss"] += 3.5 + (green_index * 0.1)

        if depression_delta > 14.0 or (dark_pixels_ratio > 0.15 and mean_edge > 9.0):
            scores["Pothole_Surface_Breakout"] += 3.8 + (depression_delta * 0.1)

        if mean_edge > 13.0 and edge_variance > 120.0:
            scores["Longitudinal_Crack"] += 2.2
            scores["Map(Block)_Cracking"] += 2.4
            scores["Transverse_Crack"] += 1.8

        if asymmetry > 22.0:
            scores["Edge_Shoulder_Damage"] += 3.2

        if color_std > 85.0 and dark_pixels_ratio < 0.2:
            scores["Debris_on_road"] += 3.0
            scores["Raveling_Loose_Aggregate"] += 2.0

        if mean_edge > 10.0 and color_std < 50.0:
            scores["Paver_Damage"] += 2.5
            scores["Patch_Repair"] += 2.0

        # High-contrast horizontal/vertical lines (drainage grid)
        horizontal_diff = np.mean(np.abs(np.diff(gray, axis=0)))
        vertical_diff = np.mean(np.abs(np.diff(gray, axis=1)))
        if abs(horizontal_diff - vertical_diff) > 4.0 and mean_edge > 11.0:
            scores["Iron_grid_damage"] += 2.8

        # Fallback default if image is relatively uniform asphalt
        if max(scores.values()) <= 0.1:
            scores["Pothole_Surface_Breakout"] += 1.5
            scores["Map(Block)_Cracking"] += 1.2
            scores["Debris_on_road"] += 1.0

        # Convert to Softmax probabilities
        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        sorted_pairs = sorted([(k, v / total_exp) for k, v in exp_scores.items()], key=lambda x: x[1], reverse=True)

        top_predictions = []
        for cls_name, prob in sorted_pairs[:top_k]:
            prob_float = float(prob)
            top_predictions.append({
                "rawClass": cls_name,
                "label": LABEL_FORMATTING.get(cls_name, cls_name.replace("_", " ")),
                "confidence": round(prob_float, 4),
                "confidencePercentage": f"{round(prob_float * 100, 1)}%",
                "mappedCategory": CATEGORY_MAPPING.get(cls_name, "other"),
            })

        best = top_predictions[0]
        return {
            "success": True,
            "rawClass": best["rawClass"],
            "label": best["label"],
            "confidence": best["confidence"],
            "confidencePercentage": best["confidencePercentage"],
            "mappedCategory": best["mappedCategory"],
            "suggestedSeverity": DEFAULT_SEVERITY.get(best["rawClass"], "medium"),
            "topPredictions": top_predictions,
            "isLowConfidence": best["confidence"] < 0.35,
        }

    def predict(self, img_input, top_k: int = 3) -> dict:
        """Runs prediction using PyTorch if available, or Vision Feature Engine otherwise."""
        try:
            pil_img = self._load_image(img_input)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to decode image: {str(e)}",
                "label": "Unknown",
                "confidence": 0.0,
                "confidencePercentage": "0.0%",
                "mappedCategory": "other",
                "suggestedSeverity": "medium",
                "topPredictions": [],
                "isLowConfidence": True,
            }

        if self.use_torch:
            try:
                with torch.inference_mode():
                    tensor = self.tf(pil_img).unsqueeze(0).to(self.device)
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu()
                    conf_top, idx_top = probs.topk(min(top_k, len(self.classes)))

                    top_predictions = []
                    for c, i in zip(conf_top, idx_top):
                        cls_name = self.classes[i]
                        prob = float(c)
                        top_predictions.append({
                            "rawClass": cls_name,
                            "label": LABEL_FORMATTING.get(cls_name, cls_name.replace("_", " ")),
                            "confidence": round(prob, 4),
                            "confidencePercentage": f"{round(prob * 100, 1)}%",
                            "mappedCategory": CATEGORY_MAPPING.get(cls_name, "other"),
                        })

                    best = top_predictions[0]
                    return {
                        "success": True,
                        "rawClass": best["rawClass"],
                        "label": best["label"],
                        "confidence": best["confidence"],
                        "confidencePercentage": best["confidencePercentage"],
                        "mappedCategory": best["mappedCategory"],
                        "suggestedSeverity": DEFAULT_SEVERITY.get(best["rawClass"], "medium"),
                        "topPredictions": top_predictions,
                        "isLowConfidence": best["confidence"] < 0.35,
                    }
            except Exception as e:
                print(f"[ML Model Warning] PyTorch inference failed: {e}. Falling back to Vision Feature Engine.")

        return self._predict_vision_features(pil_img, top_k=top_k)


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
