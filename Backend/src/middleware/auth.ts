import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import User from '../models/User';

interface JwtPayload {
  id: string;
}

declare global {
  namespace Express {
    interface Request {
      user?: any;
    }
  }
}

export const protect = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    let token;

    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }

    if (!token) {
      res.status(401).json({ success: false, error: 'Not authorized to access this route' });
      return;
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as JwtPayload;

    req.user = await User.findById(decoded.id);

    if (!req.user) {
      res.status(401).json({ success: false, error: 'User no longer exists' });
      return;
    }

    next();
  } catch (error) {
    res.status(401).json({ success: false, error: 'Not authorized to access this route' });
  }
};

export const optionalAuth = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    let token;
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    if (token) {
      const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as JwtPayload;
      req.user = await User.findById(decoded.id);
    }
    next();
  } catch (error) {
    // Non-fatal for public reporting
    next();
  }
};

export const authorize = (...roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!req.user || !roles.includes(req.user.role)) {
      res.status(403).json({ 
        success: false, 
        error: `User role ${req.user?.role} is not authorized to access this route` 
      });
      return;
    }
    next();
  };
};
