import io
import os
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pipeline import PCE_FACTORS, ANPRPipeline, load_image, public

router = APIRouter(tags=["Vehicle & License Plate ANPR"])

MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
pipe = None

def get_pipe() -> ANPRPipeline:
    global pipe
    if pipe is None:
        pipe = ANPRPipeline(use_ocr=True)
    return pipe

def _image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        return load_image(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"'{upload.filename}' is not a valid image file")

@router.get("/anpr/classes")
def classes():
    p = get_pipe()
    return {
        "vehicle_classes": list(p.vehicle.names.values()) if p.vehicle else [],
        "plate_classes": list(p.plate.names.values()) if p.plate else [],
        "pce_factors": PCE_FACTORS,
    }

@router.post("/vehicles")
def vehicles(file: UploadFile = File(...), conf: float = Query(None, ge=0.01, le=0.99)):
    p = get_pipe()
    if p.vehicle is None:
        raise HTTPException(503, "Vehicle model not loaded.")
    res = p.analyze(_image(file), vehicle_conf=conf)
    return {"filename": file.filename, **{k: res[k] for k in ("image_size", "vehicle_count", "counts", "pce_total")},
            "vehicles": [{k: v[k] for k in ("class", "confidence", "box", "pce")} for v in res["vehicles"]]}

@router.post("/plates")
def plates(file: UploadFile = File(...), conf: float = Query(None, ge=0.01, le=0.99)):
    p = get_pipe()
    if p.plate is None:
        raise HTTPException(503, "Plate model not loaded.")
    found = p.plates_with_text(_image(file), conf, read_text=False)
    return {"filename": file.filename, "plate_count": len(found),
            "plates": [{k: p[k] for k in ("class", "confidence", "box")} for p in found]}

@router.post("/anpr")
def anpr(file: UploadFile = File(...), conf: float = Query(None, ge=0.01, le=0.99)):
    p = get_pipe()
    if p.plate is None:
        raise HTTPException(503, "Plate model not loaded.")
    found = p.plates_with_text(_image(file), conf)
    return {"filename": file.filename, "plate_count": len(found),
            "plates": [{k: v for k, v in item.items() if k != "crop"} for item in found]}

@router.post("/analyze")
def analyze(file: UploadFile = File(...),
            vehicle_conf: float = Query(None, ge=0.01, le=0.99),
            plate_conf: float = Query(None, ge=0.01, le=0.99)):
    p = get_pipe()
    if p.vehicle is None:
        raise HTTPException(503, "Vehicle model not loaded.")
    res = p.analyze(_image(file), vehicle_conf, plate_conf)
    return {"filename": file.filename, **public(res)}

@router.post("/analyze/image")
def analyze_image(file: UploadFile = File(...)):
    p = get_pipe()
    if p.vehicle is None:
        raise HTTPException(503, "Vehicle model not loaded.")
    img = _image(file)
    out = p.draw(img, p.analyze(img))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
