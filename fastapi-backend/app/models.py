from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class RawTelemetryPayload(BaseModel):
    """
    Periodic raw telemetry summary sent by the Raspberry Pi 4 edge state machine
    every 1, 5, or 15 minutes.
    """
    device_id: str = Field(..., description="Unique edge node identifier e.g. 'RPI4_NODE_01'")
    window_start: Optional[str] = Field(None, description="ISO-8601 start timestamp")
    window_end: Optional[str] = Field(None, description="ISO-8601 end timestamp")
    time_window: Optional[str] = Field(None, description="Human readable or combined interval")
    raw_vehicle_count: int = Field(..., ge=0, description="Total unique vehicle passes recorded")
    speed_array: List[float] = Field(default_factory=list, description="Array of detected vehicle speeds in km/h")
    approaching_count: Optional[int] = Field(None, ge=0, description="Vehicles moving toward radar")
    receding_count: Optional[int] = Field(None, ge=0, description="Vehicles moving away from radar")
    radar_occupancy_pct: Optional[float] = Field(0.0, ge=0.0, le=100.0, description="% of time radar detected target")
    posted_speed_limit: Optional[float] = Field(35.0, description="Posted road limit in km/h")
    road_name: Optional[str] = Field("North Campus Main Road", description="Campus road segment name")
    latitude: Optional[float] = Field(31.7807, description="GPS Latitude")
    longitude: Optional[float] = Field(76.9967, description="GPS Longitude")

class ProcessedTrafficAnalytics(BaseModel):
    """
    Comprehensive civil engineering & road safety statistical profile calculated
    by the central FastAPI server.
    """
    # 1. Traffic Volume & Flow Rate
    vehicle_count: int
    flow_rate_vph: float = Field(..., description="Projected vehicles-per-hour rate")
    approaching_count: int
    receding_count: int

    # 2. Velocity & Speed Profile Analytics
    average_speed_kmh: float = Field(..., description="Mean velocity (V_avg)")
    speed_85th_percentile: float = Field(..., description="85th percentile speed (V85) for engineering evaluation")
    median_speed_kmh: float = Field(..., description="50th percentile baseline speed (V50)")
    speed_std_dev: float = Field(..., description="Speed variance / standard deviation")
    max_speed_kmh: float
    min_speed_kmh: float

    # 3. Safety Compliance & Violation Ratios
    posted_limit_kmh: float
    overspeed_count: int = Field(..., description="Vehicles exceeding posted threshold")
    violation_rate_percentage: float = Field(..., description="% of traffic violating speed limit")
    extreme_overspeed_count: int = Field(..., description="Severe reckless drivers (> 30 km/h over limit)")

    # 4. Congestion & Road Density Index
    congestion_status: str = Field(..., description="Free Flow, Moderate, Heavy Congestion, or Gridlock")
    radar_occupancy_pct: float
    congestion_index: float = Field(..., description="V_avg / V_posted ratio")
    fine_suppression_active: bool = Field(..., description="True if speed fines suppressed due to gridlock (< 0.4)")

    # 5. Temporal & Spatial Context
    device_id: str
    road_name: str
    window_start: str
    window_end: str
    window_duration_seconds: float
    is_peak_hour: bool
    calculated_at: str
    latitude: float
    longitude: float

class SpeedBreachPayload(BaseModel):
    """
    Real-time speed breach event sent when radar detects V >= V_threshold.
    Carries full 1080p JPEG snapshot taken directly from camera RAM.
    """
    device_id: str
    timestamp: Optional[str] = None
    speed_kmh: float
    posted_limit_kmh: Optional[float] = 35.0
    direction: Optional[str] = "approaching"
    image_base64: Optional[str] = None
    road_name: Optional[str] = "North Campus Main Road"
    latitude: Optional[float] = 31.7807
    longitude: Optional[float] = 76.9967

class ProcessedBreachRecord(BaseModel):
    """
    Processed violation record enriched with ML vehicle classification & license plate OCR.
    """
    breach_id: str
    device_id: str
    timestamp: str
    speed_kmh: float
    posted_limit_kmh: float
    delta_over_limit: float
    severity: str
    vehicle_type: str
    pce_weight: float
    license_plate: str
    ocr_confidence: float
    routed_to_active_learning: bool
    image_url: Optional[str] = None
    fine_suppressed: bool
    road_name: str
    latitude: float
    longitude: float

class ActiveLearningSample(BaseModel):
    """
    Low-confidence prediction (<75%) queued for server-side model retraining.
    """
    sample_id: str
    timestamp: str
    device_id: str
    image_path: str
    predicted_plate: str
    confidence: float
    vehicle_type: str
    flag_reason: str
