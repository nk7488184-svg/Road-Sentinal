import { Request, Response, NextFunction } from 'express';
import { validationResult, ValidationChain, body } from 'express-validator';

export const validate = (validations: ValidationChain[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    await Promise.all(validations.map((validation) => validation.run(req)));

    const errors = validationResult(req);
    if (errors.isEmpty()) {
      return next();
    }

    res.status(400).json({
      success: false,
      errors: errors.array()
    });
  };
};

export const reportValidation = [
  body('title').isString().isLength({ min: 3 }).withMessage('Title is required and must be at least 3 characters'),
  body('description').notEmpty().withMessage('Description is required'),
  body('category').isIn([
    'Broken_Staircase',
    'Debris_on_road',
    'Edge_Shoulder_Damage',
    'Iron_grid_damage',
    'Longitudinal_Crack',
    'Map(Block)_Cracking',
    'Patch_Repair',
    'Paver_Damage',
    'Pothole_Surface_Breakout',
    'Raveling_Loose_Aggregate',
    'Rubble_Unpaved_surface',
    'Transverse_Crack',
    'Vegetation_Moss',
    // Legacy categories for backward compatibility
    'pothole',
    'damaged_pavement',
    'faded_markings',
    'roadside_obstruction',
    'drainage_issue',
    'poor_lighting',
    'driver_behavior',
    'other'
  ]).withMessage('Invalid category'),
  body('severity').isIn(['low', 'medium', 'high', 'critical']).withMessage('Invalid severity'),
  body('location.lat').isNumeric().withMessage('Location latitude is required and must be a number'),
  body('location.lng').isNumeric().withMessage('Location longitude is required and must be a number'),
  body('location.address').optional().isString()
];

export const authValidation = {
  register: [
    body('name').notEmpty().withMessage('Name is required'),
    body('email').isEmail().withMessage('Please include a valid email'),
    body('password').isLength({ min: 6 }).withMessage('Please enter a password with 6 or more characters')
  ],
  login: [
    body('email').notEmpty().withMessage('Email is required'),
    body('password').notEmpty().withMessage('Password is required')
  ]
};
