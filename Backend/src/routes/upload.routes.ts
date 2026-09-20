import { Router } from 'express';
import multer from 'multer';
import path from 'path';
import { uploadImage, uploadMultipleImages } from '../controllers/upload.controller';
import { protect } from '../middleware/auth';

const router = Router();

const storage = multer.diskStorage({
  destination(req, file, cb) {
    cb(null, 'uploads/');
  },
  filename(req, file, cb) {
    cb(null, `${Date.now()}${path.extname(file.originalname)}`);
  }
});

const fileFilter = (req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedFileTypes = /jpeg|jpg|png|gif|webp/;
  const extname = allowedFileTypes.test(path.extname(file.originalname).toLowerCase());
  const mimetype = allowedFileTypes.test(file.mimetype);

  if (extname && mimetype) {
    return cb(null, true);
  } else {
    cb(new Error('Images only!'));
  }
};

const upload = multer({
  storage,
  fileFilter
});

router.post('/', protect, upload.single('image'), uploadImage);
router.post('/multiple', protect, upload.array('images', 5), uploadMultipleImages);

export default router;
