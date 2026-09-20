import { Router } from 'express';
import { register, login, getMe, updateProfile } from '../controllers/auth.controller';
import { protect } from '../middleware/auth';
import { validate, authValidation } from '../middleware/validate';

const router = Router();

router.post('/register', validate(authValidation.register), register);
router.post('/login', validate(authValidation.login), login);
router.get('/me', protect, getMe);
router.put('/me', protect, updateProfile);

export default router;
