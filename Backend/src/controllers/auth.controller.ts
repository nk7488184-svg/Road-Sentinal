import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import User from '../models/User';

const generateToken = (id: string): string => {
  return jwt.sign({ id }, process.env.JWT_SECRET as string, {
    expiresIn: (process.env.JWT_EXPIRE || '30d') as any
  });
};

/**
 * @desc    Register user
 * @route   POST /api/auth/register
 * @access  Public
 */
export const register = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { name, email, password } = req.body;

    const user = await User.create({
      name,
      email,
      password
    });

    const token = generateToken(user._id.toString());

    res.status(201).json({
      success: true,
      data: {
        token,
        user
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Login user
 * @route   POST /api/auth/login
 * @access  Public
 */
export const login = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email }).select('+password');

    if (!user) {
      res.status(401).json({ success: false, error: 'Invalid credentials' });
      return;
    }

    const isMatch = await user.matchPassword(password);

    if (!isMatch) {
      res.status(401).json({ success: false, error: 'Invalid credentials' });
      return;
    }

    const token = generateToken(user._id.toString());

    // Remove password from response
    user.password = undefined;

    res.status(200).json({
      success: true,
      data: {
        token,
        user
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get current logged in user
 * @route   GET /api/auth/me
 * @access  Private
 */
export const getMe = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    res.status(200).json({
      success: true,
      data: {
        user: req.user
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Update user profile
 * @route   PUT /api/auth/me
 * @access  Private
 */
export const updateProfile = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { name, email } = req.body;

    const user = await User.findByIdAndUpdate(
      req.user.id,
      { name, email },
      { new: true, runValidators: true }
    );

    res.status(200).json({
      success: true,
      data: {
        user
      }
    });
  } catch (error) {
    next(error);
  }
};
