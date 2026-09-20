import { Router } from 'express';
import { 
    getStats, 
    getTrend, 
    getCategories, 
    getHotspots, 
    getPriorityReports 
} from '../controllers/dashboard.controller';

const router = Router();

// All dashboard routes are public
router.get('/stats', getStats);
router.get('/trend', getTrend);
router.get('/categories', getCategories);
router.get('/hotspots', getHotspots);
router.get('/priority', getPriorityReports);

export default router;
