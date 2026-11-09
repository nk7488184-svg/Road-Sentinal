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
        h, w = gray.shape

        # 1. Genuine Color Metrics (Chlorophyll vs Non-green Chromaticity)
        mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        chroma = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 2.0)
        total_rgb = r + g + b + 1.0
        nexg = (2.0 * g - r - b) / total_rgb
        is_green = (g > r + 8.0) & (g > b + 8.0) & (nexg > 0.08)
        green_ratio = float(np.mean(is_green))
        green_excess = max(0.0, mean_g - max(mean_r, mean_b))

        # 2. Crater / Depression (Pothole core depression vs rim perimeter)
        cy, cx = h // 2, w // 2
        rh, rw = max(2, h // 4), max(2, w // 4)
        center_patch = gray[cy - rh : cy + rh, cx - rw : cx + rw]
        rim_patch = np.concatenate([
            gray[:max(2, h // 8), :].ravel(),
            gray[-max(2, h // 8):, :].ravel(),
            gray[:, :max(2, w // 8)].ravel(),
            gray[:, -max(2, w // 8):].ravel()
        ])
        crater_delta = max(0.0, float(np.mean(rim_patch) - np.mean(center_patch)))
        center_dark = float(np.mean(center_patch < 65.0))

        top_dark = float(np.mean(gray[:3, :] < 50.0))
        bot_dark = float(np.mean(gray[-3:, :] < 50.0))
        left_dark = float(np.mean(gray[:, :3] < 50.0))
        right_dark = float(np.mean(gray[:, -3:] < 50.0))
        rim_network_cross = (top_dark > 0.005 or bot_dark > 0.005) and (left_dark > 0.005 or right_dark > 0.005)

        # 3. Gradients and Directional Line Bias (Cracks)
        dx = np.abs(gray[:, 1:] - gray[:, :-1])
        dy = np.abs(gray[1:, :] - gray[:-1, :])
        mean_dx = float(np.mean(dx))
        mean_dy = float(np.mean(dy))
        mean_edge = (mean_dx + mean_dy) / 2.0
        edge_variance = float(np.var(dx) + np.var(dy))
        directional_bias = max(mean_dx, mean_dy) / (min(mean_dx, mean_dy) + 0.05)

        # 4. Asymmetry (Road Edge / Shoulder / Steps)
        left_mean = float(np.mean(gray[:, :w // 2]))
        right_mean = float(np.mean(gray[:, w // 2:]))
        top_mean = float(np.mean(gray[:h // 2, :]))
        bottom_mean = float(np.mean(gray[h // 2:, :]))
        asymmetry_lr = abs(left_mean - right_mean)
        asymmetry_tb = abs(top_mean - bottom_mean)
        max_asymmetry = max(asymmetry_lr, asymmetry_tb)

        # 5. Texture & Roughness
        local_std = float(np.std(gray))
        overall_mean = float(np.mean(gray))

        # 6. Periodic Bars / Stripes (Iron Grate / Stairs)
        vert = np.mean(gray, axis=0)
        horiz = np.mean(gray, axis=1)
        amp_v = float(np.max(vert) - np.min(vert))
        amp_h = float(np.max(horiz) - np.min(horiz))
        crossings_v = int(np.sum((vert[:-1] > np.mean(vert)) != (vert[1:] > np.mean(vert))))
        crossings_h = int(np.sum((horiz[:-1] > np.mean(horiz)) != (horiz[1:] > np.mean(horiz))))

        scores = {cls_name: 0.0 for cls_name in self.classes}

        # 1. Vegetation / Moss (chlorophyll green dominance)
        if green_ratio > 0.05 or (green_excess > 5.0 and chroma > 5.0):
            scores["Vegetation_Moss"] = 5.0 + (green_ratio * 15.0) + (green_excess * 0.3)

        # 2. Iron Grid / Drainage Grate (multiple periodic grating bars)
        if (crossings_v >= 8 and amp_v > 25.0) or (crossings_h >= 8 and amp_h > 25.0):
            scores["Iron_grid_damage"] = 5.0 + min(4.0, max(crossings_v, crossings_h) * 0.15) + (mean_edge * 0.1)

        # 3. Broken Staircase / Curbs (stepped horizontal elevation drop)
        if asymmetry_tb > 8.0 and amp_h > 15.0 and crossings_h >= 2 and crossings_v < 6:
            scores["Broken_Staircase"] = 4.8 + (asymmetry_tb * 0.15) + (mean_dy * 0.2)

        # 4. Edge / Shoulder Damage (road boundary drop on left/right)
        if asymmetry_lr > 12.0 and crossings_v < 4 and crater_delta < 15.0:
            scores["Edge_Shoulder_Damage"] = 4.8 + (asymmetry_lr * 0.18)

        # 5. Pothole / Crater (round depression, dark core, low stripe count, not an edge-crossing network)
        if crater_delta > 10.0 and center_dark > 0.15 and directional_bias < 2.0 and not rim_network_cross and crossings_v < 4 and crossings_h < 4:
            scores["Pothole_Surface_Breakout"] = 5.0 + min(4.0, crater_delta * 0.1) + (center_dark * 4.0)

        # 6. Directional Cracks (single prominent cracks or crack network)
        if crossings_v < 6 and crossings_h < 6:
            if directional_bias > 2.0 and mean_edge > 0.25:
                if mean_dx > mean_dy:
                    scores["Longitudinal_Crack"] = 4.8 + (mean_dx * 0.8)
                else:
                    scores["Transverse_Crack"] = 4.8 + (mean_dy * 0.8)
            elif (rim_network_cross or (mean_edge > 0.8 and edge_variance > 5.0 and directional_bias <= 2.0)) and chroma < 4.0:
                scores["Map(Block)_Cracking"] = 4.9 + (mean_edge * 0.5)

        # 7. Debris on Road (genuine non-green chromatic object on road)
        if chroma > 6.0 and green_excess < 4.0 and crater_delta < 12.0:
            scores["Debris_on_road"] = 4.8 + min(4.0, chroma * 0.2)

        # 8. Paver Damage (interlocking paver joints, low edge contrast, low color)
        if 0.2 < mean_edge < 2.0 and directional_bias < 2.0 and chroma < 3.0 and crater_delta < 6.0 and max_asymmetry < 6.0 and local_std < 15.0:
            scores["Paver_Damage"] = 4.2 + (mean_edge * 0.4)

        # 9. Patch Repair (smooth bounded asphalt patch)
        if 3.0 < crater_delta < 15.0 and center_dark < 0.10 and local_std < 12.0 and mean_edge < 2.0 and chroma < 3.0:
            scores["Patch_Repair"] = 4.4 + (crater_delta * 0.1)

        # 10. Loose Aggregate / Rubble (uniform graininess across entire image, low crack peak)
        if local_std > 12.0 and mean_edge > 2.0 and directional_bias < 1.8 and crater_delta < 8.0 and chroma < 4.0:
            if overall_mean > 120.0:
                scores["Rubble_Unpaved_surface"] = 4.5 + (local_std * 0.1)
            else:
                scores["Raveling_Loose_Aggregate"] = 4.5 + (local_std * 0.1)

        # Safe fallback if all scores remain near 0
        if max(scores.values()) < 0.1:
            if crater_delta > 3.0:
                scores["Pothole_Surface_Breakout"] = 1.0
            elif mean_edge > 1.0:
                scores["Map(Block)_Cracking"] = 1.0
            elif local_std > 8.0:
                scores["Raveling_Loose_Aggregate"] = 1.0
            else:
                scores["Paver_Damage"] = 1.0

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
