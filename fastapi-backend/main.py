import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.telemetry import router as telemetry_router
from app.routes.predict import router as predict_router, get_classifier
from app.routes.anpr import router as anpr_router
from app.websocket_manager import manager
from app.database import db

app = FastAPI(
    title="RoadSentinel Traffic Telemetry & Enforcement Engine",
    description="Central FastAPI high-throughput ingestion & statistical analytics backend for Raspberry Pi 4 roadside sensing nodes + PyTorch Road Hazard Classification ML Engine.",
    version="2.1.0",
)

# Pre-load ML model during application startup
@app.on_event("startup")
async def startup_event():
    try:
        print("[Startup] Initializing Road Hazard Classifier ML Model...")
        get_classifier()
        print("[Startup] Road Hazard Classifier ML Model ready for inference.")
    except Exception as e:
        print(f"[Startup Warning] Could not warm-load ML classifier: {e}")

# Enable CORS for Next.js Website & Dashboard frontends
app.use_all_cors = True
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence storage directory for vehicle violation snapshots
evidence_dir = os.path.join(os.path.dirname(__file__), "storage", "evidence")
os.makedirs(evidence_dir, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=evidence_dir), name="evidence")

# Register Routers
app.include_router(telemetry_router)
app.include_router(predict_router)
app.include_router(anpr_router)

@app.get("/")
def root():
    return {
        "system": "RoadSentinel Traffic Analytics & Enforcement Engine",
        "status": "online",
        "version": "2.1.0",
        "docs_url": "/docs",
        "endpoints": {
            "ml_hazard_predict": "POST /api/v1/predict",
            "ml_model_info": "GET /api/v1/predict/info",
            "telemetry_summary": "POST /api/v1/telemetry/summary",
            "speed_breach_event": "POST /api/v1/telemetry/breach",
            "latest_analytics": "GET /api/v1/telemetry/latest",
            "historical_timeseries": "GET /api/v1/telemetry/history",
            "kpi_stats": "GET /api/v1/telemetry/stats",
            "active_breaches": "GET /api/v1/telemetry/breaches",
            "active_learning_queue": "GET /api/v1/telemetry/active-learning",
            "edge_nodes": "GET /api/v1/telemetry/nodes",
            "websocket_live_feed": "ws://localhost:8000/ws/telemetry"
        }
    }

@app.get("/health")
def healthcheck():
    try:
        clf = get_classifier()
        num_classes = len(clf.classes)
        device = str(clf.device)
    except Exception:
        num_classes = 13
        device = "unknown"
    return {
        "status": "ok",
        "num_classes": num_classes,
        "device": device,
        "active_nodes": len(db.nodes),
        "telemetry_windows_logged": len(db.telemetry_history),
        "breaches_recorded": len(db.breaches)
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint.
    Broadcasts live updates whenever a Pi edge node sends a periodic summary
    or triggers a speed breach alert.
    """
    await manager.connect(websocket)
    try:
        # Send initial snapshot upon connection
        latest = db.get_latest_telemetry()
        stats = db.get_summary_stats()
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "stats": stats,
            "latest": latest.model_dump() if latest else None
        })

        while True:
            # Keep connection alive & handle incoming pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
