import { Request, Response, NextFunction } from 'express';

/**
 * @desc    Upload a single image
 * @route   POST /api/upload
 * @access  Private
 */
export const uploadImage = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ success: false, error: 'Please upload an image' });
      return;
    }

    res.status(200).json({
      success: true,
      data: {
        url: `/uploads/${req.file.filename}`
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Upload multiple images
 * @route   POST /api/upload/multiple
 * @access  Private
 */
export const uploadMultipleImages = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    if (!req.files || (req.files as Express.Multer.File[]).length === 0) {
      res.status(400).json({ success: false, error: 'Please upload at least one image' });
      return;
    }

    const files = req.files as Express.Multer.File[];
    const urls = files.map(file => `/uploads/${file.filename}`);

    res.status(200).json({
      success: true,
      data: urls
    });
  } catch (error) {
    next(error);
  }
};
