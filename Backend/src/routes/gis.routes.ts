import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';

const router = Router();
const gisDir = path.join(__dirname, '../../public/gis');

// Get all layers summary or full collection
router.get('/', (req: Request, res: Response) => {
  try {
    const summaryPath = path.join(gisDir, 'campus_all_gis.json');
    if (fs.existsSync(summaryPath)) {
      const data = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
      return res.status(200).json({
        success: true,
        data: {
          categories: Object.keys(data),
          featureCounts: {
            roads: data.roads?.features?.length || 0,
            pedestrian_paths: data.pedestrian_paths?.features?.length || 0,
            buildings: data.buildings?.features?.length || 0,
            grounds: data.grounds?.features?.length || 0,
            boundaries: data.boundaries?.features?.length || 0,
          },
          layers: data
        }
      });
    }
    return res.status(404).json({ success: false, message: 'GIS data not compiled yet' });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// Get specific category layer: roads, buildings, pedestrian_paths, grounds, boundaries
router.get('/:layer', (req: Request, res: Response) => {
  try {
    const layerName = req.params.layer.toLowerCase();
    const filePath = path.join(gisDir, `campus_${layerName}.geojson`);
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      return res.status(200).json(data);
    }
    return res.status(404).json({ success: false, message: `Layer '${layerName}' not found` });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
