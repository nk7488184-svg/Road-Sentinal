import express, { Express, Request, Response, NextFunction } from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import morgan from 'morgan';
import path from 'path';

// Load environment variables
dotenv.config();

// Import DB Connection
import connectDB from './config/db';

// Connect to database
connectDB();

// Initialize express app
const app: Express = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ limit: '25mb', extended: true }));
app.use(morgan('dev'));

// Serve static files from 'uploads' directory
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

import authRoutes from './routes/auth.routes';
import reportRoutes from './routes/report.routes';
import dashboardRoutes from './routes/dashboard.routes';
import uploadRoutes from './routes/upload.routes';

// API Welcome & Healthcheck Explorer
app.get(['/', '/api'], (req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    message: '🚀 ROAD-SENTINEL Campus Road Safety API is online and operational!',
    version: '1.0.0',
    endpoints: {
      reports: {
        allReports: 'GET /api/reports',
        nearbyReports: 'GET /api/reports/nearby?lat=31.7754&lng=76.9861',
        singleReport: 'GET /api/reports/:id',
        submitHazard: 'POST /api/reports'
      },
      dashboard: {
        kpiStats: 'GET /api/dashboard/stats',
        trends: 'GET /api/dashboard/trend',
        categories: 'GET /api/dashboard/categories',
        hotspots: 'GET /api/dashboard/hotspots',
        priorityQueue: 'GET /api/dashboard/priority'
      },
      auth: {
        login: 'POST /api/auth/login',
        register: 'POST /api/auth/register',
        profile: 'GET /api/auth/me'
      }
    }
  });
});

// Mount routes
app.use('/api/auth', authRoutes);
app.use('/api/reports', reportRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/upload', uploadRoutes);

// Error Handler Middleware
const errorHandler = (err: any, req: Request, res: Response, next: NextFunction) => {
  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
  res.status(statusCode).json({
    success: false,
    error: err.message,
    stack: process.env.NODE_ENV === 'production' ? null : err.stack,
  });
};

// Use Error Handler
app.use(errorHandler);

const DEFAULT_PORT = Number(process.env.PORT) || 3001;

function startServer(port: number) {
  const server = app.listen(port, () => {
    console.log(`\n======================================================`);
    console.log(`🚀 ROAD-SENTINEL Backend running on port ${port}`);
    console.log(`📡 URL: http://localhost:${port}/api`);
    console.log(`======================================================\n`);
  });

  server.on('error', (err: any) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`⚠️ Port ${port} is currently in use. Trying port ${port + 1}...`);
      startServer(port + 1);
    } else {
      console.error('Server error:', err);
    }
  });
}

startServer(DEFAULT_PORT);
