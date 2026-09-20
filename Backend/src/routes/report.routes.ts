import { Router } from 'express';
import { 
    getReports, 
    getNearbyReports, 
    getReport, 
    createReport, 
    updateReport, 
    deleteReport 
} from '../controllers/report.controller';
import { protect, authorize, optionalAuth } from '../middleware/auth';
import { reportValidation, validate } from '../middleware/validate';

const router = Router();

// Public routes
router.get('/', getReports);
router.get('/nearby', getNearbyReports);
router.get('/:id', getReport);

// Report submission and updates
router.post('/', optionalAuth, validate(reportValidation), createReport);
router.put('/:id', optionalAuth, updateReport);
router.patch('/:id', optionalAuth, updateReport);
router.delete('/:id', optionalAuth, deleteReport);

export default router;
