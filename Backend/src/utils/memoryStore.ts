export interface MockReport {
  id: string;
  title: string;
  description: string;
  category: string;
  severity: string;
  status: string;
  location: { lat: number; lng: number; address: string };
  images: string[];
  reportedBy: any;
  reportedAt: string;
  updatedAt: string;
  priorityScore: number;
}

export const initialReports: MockReport[] = [
  {
    id: "RPT-1001",
    title: "POTHOLE at Main Academic Block Road",
    description: "Deep pothole causing vehicle damage and traffic slowdown.",
    category: "pothole",
    severity: "critical",
    status: "submitted",
    location: { lat: 31.7754, lng: 76.9861, address: "Main Academic Block Road" },
    images: [],
    reportedBy: { name: "Admin User", email: "admin@roadsentinel.com" },
    reportedAt: new Date(Date.now() - 3600000 * 24 * 2).toISOString(),
    updatedAt: new Date().toISOString(),
    priorityScore: 85,
  },
  {
    id: "RPT-1002",
    title: "DAMAGED PAVEMENT near North Campus Gate",
    description: "Broken curb tiles causing pedestrian tripping risk.",
    category: "damaged_pavement",
    severity: "high",
    status: "in_progress",
    location: { lat: 31.7748, lng: 76.9855, address: "Near North Campus Gate" },
    images: [],
    reportedBy: { name: "Campus Reporter", email: "reporter@iitmandi.ac.in" },
    reportedAt: new Date(Date.now() - 3600000 * 24 * 5).toISOString(),
    updatedAt: new Date().toISOString(),
    priorityScore: 72,
  },
  {
    id: "RPT-1003",
    title: "POOR LIGHTING at Hostel Road - Block A",
    description: "Street lamp out of order; dark stretch posing safety concern at night.",
    category: "poor_lighting",
    severity: "high",
    status: "under_review",
    location: { lat: 31.7761, lng: 76.987, address: "Hostel Road - Block A" },
    images: [],
    reportedBy: { name: "Admin User", email: "admin@roadsentinel.com" },
    reportedAt: new Date(Date.now() - 3600000 * 24 * 1).toISOString(),
    updatedAt: new Date().toISOString(),
    priorityScore: 68,
  },
  {
    id: "RPT-1004",
    title: "DRAINAGE ISSUE at Library Junction",
    description: "Blocked culvert causing waterlogging across the road.",
    category: "drainage_issue",
    severity: "medium",
    status: "in_progress",
    location: { lat: 31.774, lng: 76.9848, address: "Library Junction" },
    images: [],
    reportedBy: { name: "Campus Reporter", email: "reporter@iitmandi.ac.in" },
    reportedAt: new Date(Date.now() - 3600000 * 24 * 10).toISOString(),
    updatedAt: new Date().toISOString(),
    priorityScore: 54,
  },
  {
    id: "RPT-1005",
    title: "FADED MARKINGS at Sports Complex Road",
    description: "Pedestrian zebra crossing completely faded.",
    category: "faded_markings",
    severity: "low",
    status: "resolved",
    location: { lat: 31.7769, lng: 76.9878, address: "Sports Complex Road" },
    images: [],
    reportedBy: { name: "Campus Reporter", email: "reporter@iitmandi.ac.in" },
    reportedAt: new Date(Date.now() - 3600000 * 24 * 20).toISOString(),
    updatedAt: new Date().toISOString(),
    priorityScore: 28,
  }
];

// Helper to sanitize images so huge base64 strings don't bloat JSON payloads
function sanitizeImages(images: any[]): string[] {
  if (!Array.isArray(images)) return [];
  return images.map(img => {
    if (typeof img !== 'string') return '';
    if (img.startsWith('data:image') && img.length > 50000) {
      // Return a lightweight placeholder / trimmed indicator for lists
      return 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect fill="%2322c55e" width="100" height="100"/><text fill="%23ffffff" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="12">Uploaded Photo</text></svg>';
    }
    return img;
  }).filter(Boolean);
}

class MemoryStore {
  private reports: MockReport[] = [...initialReports];

  getReports(query: any = {}) {
    let result = this.reports.map(r => ({
      ...r,
      // Strip oversized base64 for list endpoints so response is always lightning fast
      images: sanitizeImages(r.images)
    }));
    if (query.category) result = result.filter(r => r.category === query.category);
    if (query.severity) result = result.filter(r => r.severity === query.severity);
    if (query.status) result = result.filter(r => r.status === query.status);
    if (query.search) {
      const s = String(query.search).toLowerCase();
      result = result.filter(r => r.title.toLowerCase().includes(s) || r.location.address.toLowerCase().includes(s));
    }
    return result;
  }

  getById(id: string) {
    const target = String(id).trim().toLowerCase();
    const found = this.reports.find(r => 
      String(r.id).toLowerCase() === target || 
      String((r as any)._id).toLowerCase() === target
    );
    return found ? { ...found } : null;
  }

  addReport(reportData: any) {
    const id = `RPT-${Math.floor(1000 + Math.random() * 9000)}`;
    const newReport: MockReport = {
      id,
      title: reportData.title || 'Hazard Report',
      description: reportData.description || '',
      category: reportData.category || 'other',
      severity: reportData.severity || 'medium',
      status: reportData.status || 'submitted',
      location: reportData.location || { lat: 31.7754, lng: 76.9861, address: 'IIT Mandi Campus' },
      images: sanitizeImages(reportData.images || []),
      reportedBy: reportData.reportedBy || { name: 'Campus Student' },
      reportedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      priorityScore: reportData.priorityScore || 50,
    };
    this.reports.unshift(newReport);
    return newReport;
  }

  updateReport(id: string, updates: any) {
    const idx = this.reports.findIndex(r => r.id === id || (r as any)._id === id);
    if (idx === -1) return null;
    this.reports[idx] = { 
      ...this.reports[idx], 
      ...updates, 
      images: updates.images ? sanitizeImages(updates.images) : this.reports[idx].images,
      updatedAt: new Date().toISOString() 
    };
    return this.reports[idx];
  }

  getStats() {
    const totalReports = this.reports.length;
    const pending = this.reports.filter(r => r.status === 'submitted' || r.status === 'under_review').length;
    const inProgress = this.reports.filter(r => r.status === 'in_progress').length;
    const resolved = this.reports.filter(r => r.status === 'resolved').length;
    return { totalReports, pending, inProgress, resolved };
  }
}

export const memoryStore = new MemoryStore();

