from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List
from app.models import RawTelemetryPayload, SpeedBreachPayload, ProcessedTrafficAnalytics, ProcessedBreachRecord
from app.analytics import compute_traffic_analytics
from app.ml_pipeline import process_speed_breach
from app.database import db
from app.websocket_manager import manager

router = APIRouter(prefix="/api/v1/telemetry", tags=["Traffic Telemetry & Enforcement"])

@router.post("/summary", response_model=dict, summary="Ingest periodic raw telemetry from Raspberry Pi")
async def ingest_telemetry_summary(payload: RawTelemetryPayload, background_tasks: BackgroundTasks):
    """
    Called periodically (every 1, 5, or 15 mins) by the Raspberry Pi C++ daemon.
    Takes raw target vehicle counts and speed telemetry, then computes:
    1. Volume & flow rate (VPH)
    2. Velocity profiles (V_avg, V85, Median, StdDev)
    3. Safety compliance & violation ratios
    4. Congestion & Road Density Index
    5. Temporal context (peak hour, window duration)
    """
    try:
        processed = compute_traffic_analytics(payload)
        db.add_telemetry(processed)

        # Broadcast update in real-time to all connected web dashboards
        background_tasks.add_task(
            manager.broadcast,
            {"type": "TELEMETRY_UPDATE", "data": processed.model_dump()}
        )

        return {
            "success": True,
            "message": "Telemetry aggregated & civil engineering analytics computed successfully",
            "data": processed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process telemetry: {str(e)}")

@router.post("/breach", response_model=dict, summary="Ingest speed breach event with 1080p camera snapshot")
async def ingest_speed_breach(payload: SpeedBreachPayload, background_tasks: BackgroundTasks):
    """
    Triggered when Raspberry Pi radar registers a vehicle exceeding posted speed limit threshold.
    Processes the 1080p visual evidence, runs YOLOv8 classification + license plate ALPR,
    and routes low-confidence reads (<75%) to the server Active Learning Queue.
    """
    try:
        record, al_sample = process_speed_breach(payload)
        db.add_breach(record)
        if al_sample:
            db.add_active_learning(al_sample)

        # Broadcast live violation alert to connected web browsers
        background_tasks.add_task(
            manager.broadcast,
            {
                "type": "BREACH_ALERT",
                "data": record.model_dump(),
                "active_learning_triggered": record.routed_to_active_learning
            }
        )

        return {
            "success": True,
            "message": "Speed breach recorded and processed through ML vision pipeline",
            "data": record
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process breach: {str(e)}")

@router.get("/latest", summary="Get most recent traffic analytics window")
async def get_latest_telemetry(device_id: Optional[str] = None):
    item = db.get_latest_telemetry(device_id)
    if not item:
        raise HTTPException(status_code=404, detail="No telemetry available yet")
    return {"success": True, "data": item}

@router.get("/history", summary="Get historical traffic analytics timeseries")
async def get_telemetry_history(
    device_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    history = db.get_history(device_id, limit)
    return {"success": True, "count": len(history), "data": history}

@router.get("/stats", summary="Get high-level campus traffic KPI statistics")
async def get_campus_traffic_stats():
    stats = db.get_summary_stats()
    return {"success": True, "data": stats}

@router.get("/breaches", summary="Get list of recent speed violations")
async def get_recent_breaches(limit: int = Query(50, ge=1, le=200)):
    breaches = db.get_breaches(limit)
    return {"success": True, "count": len(breaches), "data": breaches}

@router.get("/active-learning", summary="Inspect low-confidence ALPR queue for model retraining")
async def get_active_learning_queue():
    queue = db.active_learning_queue
    return {
        "success": True,
        "count": len(queue),
        "threshold": "OCR Confidence < 75%",
        "data": queue
    }

@router.get("/nodes", summary="List registered Raspberry Pi edge sensing nodes")
async def get_registered_nodes():
    return {"success": True, "data": list(db.nodes.values())}
