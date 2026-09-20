export type HazardCategory = "pothole" | "damaged_pavement" | "faded_markings" | "roadside_obstruction" | "drainage_issue" | "poor_lighting" | "driver_behavior" | "other";
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
        case "pothole":
        case "drainage_issue":
            score += 15; break;
        case "damaged_pavement":
        case "poor_lighting":
            score += 12; break;
        case "roadside_obstruction": score += 10; break;
        case "faded_markings": score += 8; break;
        case "driver_behavior": score += 5; break;
        case "other": score += 3; break;
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
