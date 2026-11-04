from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from app.models import ProcessedTrafficAnalytics, ProcessedBreachRecord, ActiveLearningSample
import json
import os

class TelemetryStore:
    def __init__(self):
        self.telemetry_history: List[ProcessedTrafficAnalytics] = []
        self.breaches: List[ProcessedBreachRecord] = []
        self.active_learning_queue: List[ActiveLearningSample] = []
        self.nodes: Dict[str, dict] = {
            "RPI4_NORTH_GATE": {
                "device_id": "RPI4_NORTH_GATE",
                "name": "North Campus Main Gate Sensor",
                "location": "North Campus Gate, IIT Mandi",
                "latitude": 31.7816,
                "longitude": 76.9972,
                "posted_limit": 30.0,
                "status": "online",
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "firmware": "v2.4-c++17-radar",
                "radar_model": "HLK-LD2415H 24GHz",
                "network": "4G LTE (Tailscale 100.84.12.5)",
            },
            "RPI4_SOUTH_ACADEMIC": {
                "device_id": "RPI4_SOUTH_ACADEMIC",
                "name": "South Campus Academic Block Road",
                "location": "Near A16 / Academic Complex",
                "latitude": 31.7754,
                "longitude": 76.9861,
                "posted_limit": 35.0,
                "status": "online",
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "firmware": "v2.4-c++17-radar",
                "radar_model": "HLK-LD2415H 24GHz",
                "network": "4G LTE (Tailscale 100.84.12.6)",
            },
            "RPI4_HOSTEL_ROAD": {
                "device_id": "RPI4_HOSTEL_ROAD",
                "name": "Chandra Taal Hostel Road",
                "location": "Hostel Zone Access Loop",
                "latitude": 31.7768,
                "longitude": 76.9882,
                "posted_limit": 25.0,
                "status": "online",
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "firmware": "v2.4-c++17-radar",
                "radar_model": "HLK-LD2415H 24GHz",
                "network": "4G LTE (Tailscale 100.84.12.7)",
            }
        }
        self._seed_initial_data()

    def _seed_initial_data(self):
        """Pre-seeds realistic historical telemetry for instant dashboard visualisation."""
        now = datetime.now(timezone.utc)
        
        # 1. Seed Telemetry Summaries for the past 6 time-windows
        for i in range(6, 0, -1):
            w_start = now - timedelta(minutes=5 * i)
            w_end = now - timedelta(minutes=5 * (i - 1))
            
            # North Gate record
            self.telemetry_history.append(
                ProcessedTrafficAnalytics(
                    vehicle_count=18 + (i * 2),
                    flow_rate_vph=float(216 + (i * 24)),
                    approaching_count=11 + i,
                    receding_count=7 + i,
                    average_speed_kmh=round(27.4 + (i * 0.8), 1),
                    speed_85th_percentile=round(33.2 + (i * 0.5), 1),
                    median_speed_kmh=round(26.8, 1),
                    speed_std_dev=round(4.8, 2),
                    max_speed_kmh=round(42.5, 1),
                    min_speed_kmh=round(14.0, 1),
                    posted_limit_kmh=30.0,
                    overspeed_count=3,
                    violation_rate_percentage=12.5,
                    extreme_overspeed_count=0,
                    congestion_status="Free Flow",
                    radar_occupancy_pct=14.2,
                    congestion_index=0.91,
                    fine_suppression_active=False,
                    device_id="RPI4_NORTH_GATE",
                    road_name="North Campus Main Gate Road",
                    window_start=w_start.isoformat(),
                    window_end=w_end.isoformat(),
                    window_duration_seconds=300.0,
                    is_peak_hour=False,
                    calculated_at=w_end.isoformat(),
                    latitude=31.7816,
                    longitude=76.9972,
                )
            )

            # South Academic record
            self.telemetry_history.append(
                ProcessedTrafficAnalytics(
                    vehicle_count=24 + (i * 3),
                    flow_rate_vph=float(288 + (i * 36)),
                    approaching_count=14 + i,
                    receding_count=10 + i,
                    average_speed_kmh=round(31.2 + (i * 0.6), 1),
                    speed_85th_percentile=round(38.6, 1),
                    median_speed_kmh=round(30.5, 1),
                    speed_std_dev=round(5.4, 2),
                    max_speed_kmh=round(47.2, 1),
                    min_speed_kmh=round(18.2, 1),
                    posted_limit_kmh=35.0,
                    overspeed_count=5,
                    violation_rate_percentage=17.2,
                    extreme_overspeed_count=0,
                    congestion_status="Free Flow",
                    radar_occupancy_pct=18.5,
                    congestion_index=0.89,
                    fine_suppression_active=False,
                    device_id="RPI4_SOUTH_ACADEMIC",
                    road_name="South Campus Academic Complex Road",
                    window_start=w_start.isoformat(),
                    window_end=w_end.isoformat(),
                    window_duration_seconds=300.0,
                    is_peak_hour=False,
                    calculated_at=w_end.isoformat(),
                    latitude=31.7754,
                    longitude=76.9861,
                )
            )

        # 2. Seed Initial Speed Breach Records
        self.breaches.append(
            ProcessedBreachRecord(
                breach_id="BREACH-A801",
                device_id="RPI4_SOUTH_ACADEMIC",
                timestamp=(now - timedelta(minutes=14)).isoformat(),
                speed_kmh=51.4,
                posted_limit_kmh=35.0,
                delta_over_limit=16.4,
                severity="high",
                vehicle_type="SUV",
                pce_weight=1.2,
                license_plate="HP 33 B 4921",
                ocr_confidence=0.94,
                routed_to_active_learning=False,
                image_url=None,
                fine_suppressed=False,
                road_name="South Campus Academic Complex Road",
                latitude=31.7754,
                longitude=76.9861,
            )
        )
        self.breaches.append(
            ProcessedBreachRecord(
                breach_id="BREACH-C109",
                device_id="RPI4_NORTH_GATE",
                timestamp=(now - timedelta(minutes=42)).isoformat(),
                speed_kmh=44.8,
                posted_limit_kmh=30.0,
                delta_over_limit=14.8,
                severity="moderate",
                vehicle_type="Car",
                pce_weight=1.0,
                license_plate="HP 65 A 1084",
                ocr_confidence=0.71,
                routed_to_active_learning=True,
                image_url=None,
                fine_suppressed=False,
                road_name="North Campus Main Gate Road",
                latitude=31.7816,
                longitude=76.9972,
            )
        )

    def add_telemetry(self, item: ProcessedTrafficAnalytics):
        self.telemetry_history.append(item)
        if len(self.telemetry_history) > 500:
            self.telemetry_history.pop(0)

        # Update node status
        if item.device_id in self.nodes:
            self.nodes[item.device_id]["last_seen"] = item.calculated_at
            self.nodes[item.device_id]["status"] = "online"
        else:
            self.nodes[item.device_id] = {
                "device_id": item.device_id,
                "name": f"Edge Sensor {item.device_id}",
                "location": item.road_name,
                "latitude": item.latitude,
                "longitude": item.longitude,
                "posted_limit": item.posted_limit_kmh,
                "status": "online",
                "last_seen": item.calculated_at,
                "firmware": "v2.4-c++17-radar",
                "radar_model": "HLK-LD2415H 24GHz",
                "network": "4G LTE",
            }

    def add_breach(self, breach: ProcessedBreachRecord):
        self.breaches.insert(0, breach)
        if len(self.breaches) > 200:
            self.breaches.pop()

    def add_active_learning(self, sample: ActiveLearningSample):
        self.active_learning_queue.insert(0, sample)
        if len(self.active_learning_queue) > 100:
            self.active_learning_queue.pop()

    def get_latest_telemetry(self, device_id: Optional[str] = None) -> Optional[ProcessedTrafficAnalytics]:
        if not self.telemetry_history:
            return None
        if device_id:
            matches = [t for t in reversed(self.telemetry_history) if t.device_id == device_id]
            return matches[0] if matches else None
        return self.telemetry_history[-1]

    def get_history(self, device_id: Optional[str] = None, limit: int = 50) -> List[ProcessedTrafficAnalytics]:
        res = self.telemetry_history
        if device_id:
            res = [t for t in res if t.device_id == device_id]
        return list(reversed(res[-limit:]))

    def get_breaches(self, limit: int = 50) -> List[ProcessedBreachRecord]:
        return self.breaches[:limit]

    def get_summary_stats(self) -> dict:
        """Returns consolidated campus-wide KPI stats for UI cards."""
        latest = self.telemetry_history[-1] if self.telemetry_history else None
        total_vehicles_logged = sum(t.vehicle_count for t in self.telemetry_history[-20:])
        avg_speed_campus = round(sum(t.average_speed_kmh for t in self.telemetry_history[-10:]) / max(1, len(self.telemetry_history[-10:])), 1) if self.telemetry_history else 0.0
        
        # Overall compliance rate
        total_overspeed = sum(t.overspeed_count for t in self.telemetry_history[-20:])
        overall_violation_rate = round((total_overspeed / max(1, total_vehicles_logged)) * 100.0, 1)

        return {
            "active_nodes_count": len([n for n in self.nodes.values() if n.get("status") == "online"]),
            "total_nodes": len(self.nodes),
            "recent_vehicle_volume": total_vehicles_logged,
            "campus_avg_speed_kmh": avg_speed_campus,
            "campus_violation_rate": overall_violation_rate,
            "total_breaches_recorded": len(self.breaches),
            "active_learning_queued": len(self.active_learning_queue),
            "latest_congestion_status": latest.congestion_status if latest else "Free Flow",
            "current_flow_rate_vph": latest.flow_rate_vph if latest else 0.0,
            "v85_speed_kmh": latest.speed_85th_percentile if latest else 0.0,
        }

db = TelemetryStore()
