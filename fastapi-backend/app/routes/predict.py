from fastapi import APIRouter, Request, HTTPException
from typing import Optional, List, Dict, Any
import os
import sys

# Add parent directory to path so classifier can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from classifier import RoadHazardClassifier

router = APIRouter(prefix="/api/v1", tags=["ML Road Hazard Classification"])

# Lazy / startup loader for the classifier singleton
_classifier: Optional[RoadHazardClassifier] = None

def get_classifier() -> RoadHazardClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RoadHazardClassifier.get_instance()
    return _classifier


@router.get("/predict/info")
def get_model_info():
    """Returns model architecture, loaded status, and class label catalog."""
    try:
        clf = get_classifier()
        return {
            "model_name": "EfficientNetV2-S Road Hazard Classifier",
            "status": "ready",
            "img_size": clf.img_size,
            "classes_count": len(clf.classes),
            "classes": clf.classes,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/predict")
async def predict_hazard(request: Request):
    """
    Accepts an uploaded image via:
    1. JSON body: { "image": "data:image/jpeg;base64,..." } or { "image_base64": "..." }
    2. Multipart form: form field 'file' or 'image'
    3. Raw binary image stream
    """
    try:
        clf = get_classifier()
        content_type = request.headers.get("content-type", "").lower()
        
        # 1. Handle JSON payload
        if "application/json" in content_type:
            data = await request.json()
            img_input = data.get("image") or data.get("image_base64")
            if not img_input:
                raise HTTPException(status_code=400, detail="Missing 'image' or 'image_base64' in JSON body.")
            return clf.predict(img_input)

        # 2. Handle Multipart Form
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file") or form.get("image")
            if not file_field:
                raise HTTPException(status_code=400, detail="No file attached. Use field name 'file' or 'image'.")
            contents = await file_field.read()
            return clf.predict(contents)

        # 3. Handle raw binary or fallback to JSON/text
        else:
            raw_body = await request.body()
            if not raw_body:
                raise HTTPException(status_code=400, detail="Request body is empty.")
            # Check if JSON string
            try:
                import json
                data = json.loads(raw_body.decode("utf-8"))
                img_input = data.get("image") or data.get("image_base64")
                if img_input:
                    return clf.predict(img_input)
            except Exception:
                pass
            # Treat as raw image bytes
            return clf.predict(raw_body)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
