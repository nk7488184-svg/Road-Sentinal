import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple
from app.models import RawTelemetryPayload, ProcessedTrafficAnalytics

def parse_iso_or_now(ts_str: str | None) -> datetime:
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        # replace Z if present
        clean = ts_str.replace('Z', '+00:00')
        return datetime.fromisoformat(clean)
    except Exception:
        return datetime.now(timezone.utc)

def is_peak_hour_check(dt: datetime) -> bool:
    """
    Evaluates whether the timestamp falls into IIT Mandi campus peak rush hours:
    Morning commute: 08:00 - 10:30
    Evening commute: 17:00 - 19:30
    """
    # Convert to IST (UTC+5:30) for IIT Mandi
    ist_offset = timedelta(hours=5, minutes=30)
    ist_time = dt.astimezone(timezone.utc) + ist_offset
    hour = ist_time.hour
    minute = ist_time.minute
    total_minutes = hour * 60 + minute

    morning_peak = (8 * 60) <= total_minutes <= (10 * 60 + 30)
    evening_peak = (17 * 60) <= total_minutes <= (19 * 60 + 30)
    return morning_peak or evening_peak

def compute_traffic_analytics(payload: RawTelemetryPayload) -> ProcessedTrafficAnalytics:
    """
    Executes complete statistical telemetry analysis across the 5 core domains:
    1. Traffic Volume & Flow Rate
    2. Velocity & Speed Profile Analytics (V_avg, V85, Median, StdDev)
    3. Safety Compliance & Violation Ratios
    4. Congestion & Road Density Index (Greenshields relation)
    5. Temporal & Spatial Context
    """
    now = datetime.now(timezone.utc)
    start_dt = parse_iso_or_now(payload.window_start)
    end_dt = parse_iso_or_now(payload.window_end)

    duration_seconds = (end_dt - start_dt).total_seconds()
    if duration_seconds <= 0:
        duration_seconds = 300.0  # default 5 minute standard window

    # Count & Direction Split
    vehicle_count = max(payload.raw_vehicle_count, len(payload.speed_array))
    
    # Standardised Vehicles-Per-Hour (VPH) Flow Rate:
    # Flow Rate = Count * (3600 / Window Duration)
    window_multiplier = 3600.0 / duration_seconds
    flow_rate_vph = round(vehicle_count * window_multiplier, 1)

    if payload.approaching_count is not None and payload.receding_count is not None:
        approaching = payload.approaching_count
        receding = payload.receding_count
    else:
        # Default balanced split if directional radar feed did not specify
        approaching = int(np.ceil(vehicle_count * 0.52))
        receding = max(0, vehicle_count - approaching)

    # Velocity & Speed Profiles
    speeds = np.array(payload.speed_array, dtype=float) if payload.speed_array else np.array([])
    posted_limit = float(payload.posted_speed_limit or 35.0)

    if len(speeds) > 0:
        avg_speed = float(np.mean(speeds))
        v85 = float(np.percentile(speeds, 85))
        v50 = float(np.median(speeds))
        speed_std = float(np.std(speeds))
        max_speed = float(np.max(speeds))
        min_speed = float(np.min(speeds))

        # Overspeed & Violation counts
        overspeed_count = int(np.sum(speeds > posted_limit))
        extreme_overspeed = int(np.sum(speeds > (posted_limit + 30.0)))
        violation_rate = round((overspeed_count / len(speeds)) * 100.0, 1)
    else:
        avg_speed = 0.0
        v85 = 0.0
        v50 = 0.0
        speed_std = 0.0
        max_speed = 0.0
        min_speed = 0.0
        overspeed_count = 0
        extreme_overspeed = 0
        violation_rate = 0.0

    # Congestion & Road Density Index
    # Congestion Index = V_avg / V_posted
    occupancy = float(payload.radar_occupancy_pct or 0.0)
    
    if posted_limit > 0 and len(speeds) > 0:
        congestion_index = round(avg_speed / posted_limit, 2)
    else:
        congestion_index = 1.0

    # Categorize Congestion Status:
    # Free Flow: index >= 0.85
    # Moderate: 0.60 <= index < 0.85
    # Heavy Congestion: 0.40 <= index < 0.60
    # Gridlock: index < 0.40 (or very high occupancy with crawling speed)
    if (occupancy >= 80.0 and avg_speed < 10.0) or congestion_index < 0.40:
        congestion_status = "Gridlock"
        fine_suppression = True  # Automated suppression of speed fines during bumper-to-bumper traffic
    elif congestion_index < 0.60:
        congestion_status = "Heavy Congestion"
        fine_suppression = False
    elif congestion_index < 0.85:
        congestion_status = "Moderate"
        fine_suppression = False
    else:
        congestion_status = "Free Flow"
        fine_suppression = False

    is_peak = is_peak_hour_check(start_dt)

    return ProcessedTrafficAnalytics(
        vehicle_count=vehicle_count,
        flow_rate_vph=flow_rate_vph,
        approaching_count=approaching,
        receding_count=receding,
        average_speed_kmh=round(avg_speed, 1),
        speed_85th_percentile=round(v85, 1),
        median_speed_kmh=round(v50, 1),
        speed_std_dev=round(speed_std, 2),
        max_speed_kmh=round(max_speed, 1),
        min_speed_kmh=round(min_speed, 1),
        posted_limit_kmh=round(posted_limit, 1),
        overspeed_count=overspeed_count,
        violation_rate_percentage=violation_rate,
        extreme_overspeed_count=extreme_overspeed,
        congestion_status=congestion_status,
        radar_occupancy_pct=round(occupancy, 1),
        congestion_index=congestion_index,
        fine_suppression_active=fine_suppression,
        device_id=payload.device_id,
        road_name=payload.road_name or "Campus Road",
        window_start=start_dt.isoformat(),
        window_end=end_dt.isoformat(),
        window_duration_seconds=duration_seconds,
        is_peak_hour=is_peak,
        calculated_at=now.isoformat(),
        latitude=payload.latitude or 31.7807,
        longitude=payload.longitude or 76.9967,
    )
