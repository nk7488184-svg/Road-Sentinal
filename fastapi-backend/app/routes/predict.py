from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Query
from typing import Optional, List, Dict, Any
import os
import sys

# Add parent directory to path so classifier can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from classifier import RoadHazardClassifier

router = APIRouter(tags=["ML Road Hazard Classification"])

MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_BATCH = 20

# Lazy / startup loader for the classifier singleton
_classifier: Optional[RoadHazardClassifier] = None

def get_classifier() -> RoadHazardClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RoadHazardClassifier.get_instance()
    return _classifier


@router.get("/classes")
@router.get("/api/v1/classes")
def get_classes():
    """Returns the list of 13 supported road hazard classes."""
    clf = get_classifier()
    return {
        "num_classes": len(clf.classes),
        "classes": clf.classes
    }


@router.get("/api/v1/predict/info")
def get_model_info():
    """Returns model architecture, loaded status, and class label catalog."""
    try:
        clf = get_classifier()
        return {
            "model_name": "EfficientNetV2-S Road Hazard Classifier",
            "status": "ready",
            "device": str(clf.device),
            "img_size": clf.img_size,
            "classes_count": len(clf.classes),
            "classes": clf.classes,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/predict")
@router.post("/api/v1/predict")
async def predict_hazard(
    request: Request,
    top_k: int = Query(3, ge=1, le=20, description="How many top classes to return")
):
    """
    Accepts road hazard image via:
    1. Multipart form-data: field 'file' (RD_API standard) or 'image'
    2. JSON body: { "image": "data:image/jpeg;base64,..." } or { "image_base64": "..." }
    3. Raw binary image stream
    """
    try:
        clf = get_classifier()
        content_type = request.headers.get("content-type", "").lower()
        filename = "road.jpg"

        # 1. Handle Multipart Form
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file") or form.get("image")
            if not file_field:
                raise HTTPException(status_code=400, detail="No image file attached. Use form field 'file' or 'image'.")
            
            if hasattr(file_field, "filename") and file_field.filename:
                filename = file_field.filename

            contents = await file_field.read()
            if len(contents) > MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of {MAX_BYTES // (1024 * 1024)} MB")
            
            return clf.predict(contents, top_k=top_k, filename=filename)

        # 2. Handle JSON payload
        elif "application/json" in content_type:
            data = await request.json()
            img_input = data.get("image") or data.get("image_base64")
            if not img_input:
                raise HTTPException(status_code=400, detail="Missing 'image' or 'image_base64' in JSON body.")
            req_filename = data.get("filename", "road.jpg")
            return clf.predict(img_input, top_k=top_k, filename=req_filename)

        # 3. Handle raw binary or fallback to JSON/text
        else:
            raw_body = await request.body()
            if not raw_body:
                raise HTTPException(status_code=400, detail="Request body is empty.")
            if len(raw_body) > MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of {MAX_BYTES // (1024 * 1024)} MB")

            # Check if JSON string
            try:
                import json
                data = json.loads(raw_body.decode("utf-8"))
                img_input = data.get("image") or data.get("image_base64")
                if img_input:
                    req_filename = data.get("filename", "road.jpg")
                    return clf.predict(img_input, top_k=top_k, filename=req_filename)
            except Exception:
                pass

            # Treat as raw image bytes
            return clf.predict(raw_body, top_k=top_k, filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/predict/batch")
@router.post("/api/v1/predict/batch")
async def predict_batch(
    request: Request,
    top_k: int = Query(3, ge=1, le=20, description="How many top classes to return")
):
    """Batch prediction endpoint for up to MAX_BATCH images."""
    clf = get_classifier()
    form = await request.form()
    files = form.getlist("files") or form.getlist("file")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded in 'files' field.")
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Send at most {MAX_BATCH} images per request.")

    results = []
    for f in files:
        fname = getattr(f, "filename", "road.jpg")
        try:
            contents = await f.read()
            if len(contents) > MAX_BYTES:
                results.append({"filename": fname, "error": f"File exceeds {MAX_BYTES // (1024 * 1024)} MB"})
            else:
                results.append(clf.predict(contents, top_k=top_k, filename=fname))
        except Exception as e:
            results.append({"filename": fname, "error": str(e)})

    return {"count": len(results), "results": results}
