import os
import io
import base64
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 13 Class Names from checkpoint:
# ['Broken_Staircase', 'Debris_on_road', 'Edge_Shoulder_Damage', 'Iron_grid_damage',
#  'Longitudinal_Crack', 'Map(Block)_Cracking', 'Patch_Repair', 'Paver_Damage',
#  'Pothole_Surface_Breakout', 'Raveling_Loose_Aggregate', 'Rubble_Unpaved_surface',
#  'Transverse_Crack', 'Vegetation_Moss']

CATEGORY_MAPPING = {
    "Pothole_Surface_Breakout": "pothole",
    "Longitudinal_Crack": "damaged_pavement",
    "Transverse_Crack": "damaged_pavement",
    "Map(Block)_Cracking": "damaged_pavement",
    "Paver_Damage": "damaged_pavement",
    "Edge_Shoulder_Damage": "damaged_pavement",
    "Patch_Repair": "damaged_pavement",
    "Raveling_Loose_Aggregate": "damaged_pavement",
    "Rubble_Unpaved_surface": "damaged_pavement",
    "Broken_Staircase": "damaged_pavement",
    "Debris_on_road": "roadside_obstruction",
    "Vegetation_Moss": "roadside_obstruction",
    "Iron_grid_damage": "drainage_issue",
}

LABEL_FORMATTING = {
    "Pothole_Surface_Breakout": "Pothole / Surface Breakout",
    "Longitudinal_Crack": "Longitudinal Crack",
    "Transverse_Crack": "Transverse Crack",
    "Map(Block)_Cracking": "Map (Block) Cracking",
    "Paver_Damage": "Paver Block Damage",
    "Edge_Shoulder_Damage": "Edge / Shoulder Damage",
    "Patch_Repair": "Patch Repair Deterioration",
    "Raveling_Loose_Aggregate": "Raveling / Loose Aggregate",
    "Rubble_Unpaved_surface": "Rubble / Unpaved Surface",
    "Broken_Staircase": "Broken Staircase / Curbside",
    "Debris_on_road": "Debris on Road",
    "Vegetation_Moss": "Vegetation / Moss Overgrowth",
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
                # search candidate paths
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                candidates = [
                    os.path.join(curr_dir, "best_model.pth"),
                    os.path.join(curr_dir, "..", "fastapi-backend", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "DP-PROJECT", "fastapi-backend", "best_model.pth"),
                    os.path.join(os.path.expanduser("~"), "Downloads", "best_model.pth"),
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        model_path = c
                        break
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model checkpoint not found. Looked in: {model_path}")
            cls._instance = RoadHazardClassifier(model_path)
        return cls._instance

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ML Model] Loading checkpoint from {model_path} on {self.device}...")
        ck = torch.load(model_path, map_location=self.device)
        self.classes = ck["classes"]
        self.img_size = ck.get("img_size", 300)

        # Build architecture: EfficientNetV2-S
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(self.classes))
        )
        model.load_state_dict(ck["model"])
        self.model = model.to(self.device).eval()

        # Exact training preprocessing: Resize(300, 300), ToTensor(), Normalize
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"[ML Model] Loaded {len(self.classes)} classes successfully!")

    def _load_image(self, img_input) -> Image.Image:
        """Converts diverse image formats (PIL, bytes, base64) into PIL.Image."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        
        if isinstance(img_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_input)).convert("RGB")
        
        if isinstance(img_input, str):
            # Check if base64 data uri
            if img_input.startswith("data:image"):
                base64_data = img_input.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Raw base64 string
            try:
                img_bytes = base64.b64decode(img_input)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass
            # File path on disk
            if os.path.isfile(img_input):
                return Image.open(img_input).convert("RGB")

        raise ValueError("Unsupported image input format. Must be PIL.Image, bytes, base64 string, or valid file path.")

    @torch.inference_mode()
    def predict(self, img_input, top_k: int = 3) -> dict:
        """
        Runs inference on the provided image and returns rich structured metadata.
        """
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
        is_low_conf = best["confidence"] < 0.35

        return {
            "success": True,
            "rawClass": best["rawClass"],
            "label": best["label"],
            "confidence": best["confidence"],
            "confidencePercentage": best["confidencePercentage"],
            "mappedCategory": best["mappedCategory"],
            "suggestedSeverity": DEFAULT_SEVERITY.get(best["rawClass"], "medium"),
            "topPredictions": top_predictions,
            "isLowConfidence": is_low_conf,
        }


# Quick test or CLI fallback
if __name__ == "__main__":
    import sys
    classifier = RoadHazardClassifier.get_instance()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--stdin":
            input_data = sys.stdin.read().strip()
            res = classifier.predict(input_data)
        else:
            res = classifier.predict(arg)
        print(json.dumps(res, indent=2))
    elif not sys.stdin.isatty():
        input_data = sys.stdin.read().strip()
        if input_data:
            res = classifier.predict(input_data)
            print(json.dumps(res, indent=2))
        else:
            dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
            res = classifier.predict(dummy_img)
            print(json.dumps(res, indent=2))
    else:
        # Dummy test tensor image
        dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        res = classifier.predict(dummy_img)
        print("Model verification test result:")
        print(json.dumps(res, indent=2))
