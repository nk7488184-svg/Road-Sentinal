export type HazardCategory =
  | "Broken_Staircase"
  | "Debris_on_road"
  | "Edge_Shoulder_Damage"
  | "Iron_grid_damage"
  | "Longitudinal_Crack"
  | "Map(Block)_Cracking"
  | "Patch_Repair"
  | "Paver_Damage"
  | "Pothole_Surface_Breakout"
  | "Raveling_Loose_Aggregate"
  | "Rubble_Unpaved_surface"
  | "Transverse_Crack"
  | "Vegetation_Moss"
  | string;

export type Severity = "low" | "medium" | "high" | "critical";

/**
 * Calculates priority score from 0-100 based on severity, category, and age
 */
export function calculatePriorityScore(severity: Severity, category: HazardCategory, reportedAt: Date | string): number {
    let score = 0;
    
    // Severity weight
    switch (severity) {
        case "critical": score += 40; break;
        case "high": score += 30; break;
        case "medium": score += 20; break;
        case "low": score += 10; break;
    }

    // Category weight
    switch (category) {
        case "Pothole_Surface_Breakout":
        case "pothole":
        case "Broken_Staircase":
        case "Iron_grid_damage":
        case "drainage_issue":
        case "drainage_iron_grid":
        case "broken_curb":
            score += 15; break;
        case "Map(Block)_Cracking":
        case "Longitudinal_Crack":
        case "Transverse_Crack":
        case "Paver_Damage":
        case "Edge_Shoulder_Damage":
        case "damaged_pavement":
        case "road_cracks":
        case "edge_damage":
        case "poor_lighting":
            score += 12; break;
        case "Debris_on_road":
        case "debris_on_road":
        case "Rubble_Unpaved_surface":
        case "Patch_Repair":
        case "Raveling_Loose_Aggregate":
        case "roadside_obstruction":
        case "patch_deterioration":
        case "loose_aggregate":
            score += 10; break;
        case "Vegetation_Moss":
        case "vegetation":
        case "faded_markings":
            score += 8; break;
        default:
            score += 5; break;
    }

    // Age weight (capped at 30 days = 30 points)
    const reportDate = new Date(reportedAt);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - reportDate.getTime());
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    const ageScore = Math.min(diffDays, 30);
    score += ageScore;

    return Math.min(score, 100);
}
