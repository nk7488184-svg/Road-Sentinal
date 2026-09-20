import { Request, Response } from 'express';
import mongoose from 'mongoose';
import HazardReport from '../models/HazardReport';
import { calculatePriorityScore, Severity, HazardCategory } from '../utils/priorityCalculator';
import { memoryStore } from '../utils/memoryStore';

/**
 * Get all reports with optional filtering, search, and pagination
 */
export const getReports = async (req: Request, res: Response) => {
    try {
        const { category, severity, status, search, page = '1', limit = '50', sort = '-reportedAt' } = req.query;
        
        const pageNum = parseInt(page as string, 10) || 1;
        const limitNum = parseInt(limit as string, 10) || 50;
        const skip = (pageNum - 1) * limitNum;

        // Retrieve from fast unified memory store (includes all submissions)
        const allReports = memoryStore.getReports(req.query);
        const pagedReports = allReports.slice(skip, skip + limitNum);

        return res.status(200).json({
            success: true,
            data: pagedReports,
            pagination: {
                page: pageNum,
                limit: limitNum,
                total: allReports.length,
                pages: Math.max(1, Math.ceil(allReports.length / limitNum))
            }
        });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};

/**
 * Get single report by ID
 */
export const getReport = async (req: Request, res: Response) => {
    try {
        const targetId = req.params.id;

        // Check memory store first (fastest and always up-to-date with runtime submissions)
        const foundMem = memoryStore.getById(targetId);
        if (foundMem) {
            return res.status(200).json({ success: true, data: foundMem });
        }

        if (mongoose.connection.readyState === 1) {
            try {
                const report = await HazardReport.findById(targetId).populate('reportedBy', 'name email');
                if (report) {
                    return res.status(200).json({ success: true, data: report });
                }
            } catch (err) {
                // ignore cast error
            }
        }

        return res.status(404).json({ success: false, message: 'Report not found' });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};

/**
 * Create a new report
 */
export const createReport = async (req: Request, res: Response) => {
    try {
        const { title, description, category, severity, location, images } = req.body;
        
        const reportedAt = new Date();
        const priorityScore = calculatePriorityScore(severity as Severity, category as HazardCategory, reportedAt);
        const reportedBy = (req as any).user ? (req as any).user : { name: 'Anonymous Campus Reporter' };
        
        // 1. Immediately save into memory store
        const saved = memoryStore.addReport({
            title,
            description,
            category,
            severity,
            location,
            images: images || [],
            reportedBy,
            priorityScore
        });

        // 2. Also persist to MongoDB asynchronously if online
        if (mongoose.connection.readyState === 1) {
            HazardReport.create({
                title,
                description,
                category,
                severity,
                location,
                images: images || [],
                reportedBy: (req as any).user?.id || null,
                reportedAt,
                priorityScore,
                status: 'submitted'
            }).catch(() => {});
        }
        
        return res.status(201).json({ success: true, data: saved });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};

/**
 * Update report by ID
 */
export const updateReport = async (req: Request, res: Response) => {
    try {
        const { category, severity, status, title, description, location, images } = req.body;
        const targetId = req.params.id;

        const updatedMem = memoryStore.updateReport(targetId, {
            category, severity, status, title, description, location, images
        });

        if (mongoose.connection.readyState === 1) {
            try {
                const report = await HazardReport.findById(targetId);
                if (report) {
                    if (title) report.title = title;
                    if (description) report.description = description;
                    if (location) report.location = location;
                    if (images) report.images = images;
                    if (status) report.status = status;
                    if (category) report.category = category;
                    if (severity) report.severity = severity;
                    report.updatedAt = new Date();
                    await report.save();
                }
            } catch (err) {}
        }

        if (updatedMem) {
            return res.status(200).json({ success: true, data: updatedMem });
        }

        return res.status(404).json({ success: false, message: 'Report not found' });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};

/**
 * Delete report by ID
 */
export const deleteReport = async (req: Request, res: Response) => {
    try {
        const targetId = req.params.id;
        if (mongoose.connection.readyState === 1) {
            try {
                await HazardReport.findByIdAndDelete(targetId);
            } catch (err) {}
        }
        return res.status(200).json({ success: true, message: 'Report deleted successfully' });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};

/**
 * Get nearby reports using Haversine formula
 */
export const getNearbyReports = async (req: Request, res: Response) => {
    try {
        const { lat, lng, radius = '2' } = req.query;
        
        if (!lat || !lng) {
            return res.status(400).json({ success: false, message: 'Latitude and longitude are required' });
        }
        
        const latNum = parseFloat(lat as string);
        const lngNum = parseFloat(lng as string);
        const radiusNum = parseFloat(radius as string);
        const earthRadiusKm = 6371;
        
        const allReports = memoryStore.getReports({});
        
        const nearbyReports = allReports.filter(report => {
            if (!report.location || report.location.lat === undefined || report.location.lng === undefined) {
                return false;
            }
            
            const dLat = (report.location.lat - latNum) * (Math.PI / 180);
            const dLng = (report.location.lng - lngNum) * (Math.PI / 180);
            const a = 
                Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(latNum * (Math.PI / 180)) * Math.cos(report.location.lat * (Math.PI / 180)) * 
                Math.sin(dLng / 2) * Math.sin(dLng / 2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            const distance = earthRadiusKm * c;
            
            return distance <= radiusNum;
        });
        
        return res.status(200).json({ success: true, data: nearbyReports });
    } catch (error: any) {
        return res.status(500).json({ success: false, message: 'Server error', error: error.message });
    }
};
