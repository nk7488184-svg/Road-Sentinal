import mongoose from 'mongoose';
import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import User from '../models/User';
import HazardReport from '../models/HazardReport';
import { calculatePriorityScore, Severity, HazardCategory } from './priorityCalculator';

// Load env vars
dotenv.config();

const connectDB = async () => {
    try {
        const conn = await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/road-sentinel');
        console.log(`MongoDB Connected: ${conn.connection.host}`);
    } catch (error: any) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
};

const campusLocations = [
    { lat: 31.7754, lng: 76.9861, address: 'Main Academic Block Road' },
    { lat: 31.7748, lng: 76.9855, address: 'Near North Campus Gate' },
    { lat: 31.7761, lng: 76.9870, address: 'Hostel Road - Block A' },
    { lat: 31.7740, lng: 76.9848, address: 'Library Junction' },
    { lat: 31.7769, lng: 76.9878, address: 'Sports Complex Road' },
    { lat: 31.7735, lng: 76.9842, address: 'South Campus Entrance' },
    { lat: 31.7757, lng: 76.9865, address: 'Cafeteria Parking Lot' },
    { lat: 31.7745, lng: 76.9852, address: 'Workshop Road' },
    { lat: 31.7763, lng: 76.9873, address: 'Hostel Block C Entrance' },
    { lat: 31.7750, lng: 76.9858, address: 'Central Lawn Pathway' },
    { lat: 31.7738, lng: 76.9845, address: 'Admin Building Curve' },
    { lat: 31.7772, lng: 76.9882, address: 'Faculty Quarters Road' },
    { lat: 31.7743, lng: 76.9850, address: 'Auditorium Lane' },
    { lat: 31.7766, lng: 76.9876, address: 'Medical Center Road' },
    { lat: 31.7752, lng: 76.9863, address: 'SBI ATM Road' },
    { lat: 31.7759, lng: 76.9868, address: 'OAT Approach Road' },
    { lat: 31.7747, lng: 76.9854, address: 'Lab Complex Junction' },
    { lat: 31.7770, lng: 76.9880, address: 'Kamand Bridge Road' },
    { lat: 31.7736, lng: 76.9843, address: 'Guest House Path' },
    { lat: 31.7755, lng: 76.9862, address: 'Mess Road' },
    { lat: 31.7742, lng: 76.9849, address: 'Parking Area B' },
    { lat: 31.7768, lng: 76.9879, address: 'Tennis Court Road' },
    { lat: 31.7733, lng: 76.9840, address: 'Main Gate Approach' },
    { lat: 31.7760, lng: 76.9869, address: 'Hostel Block D Road' },
    { lat: 31.7746, lng: 76.9853, address: 'Computer Center Path' },
    { lat: 31.7758, lng: 76.9867, address: 'Research Lab Road' },
    { lat: 31.7741, lng: 76.9847, address: 'Playground Access Road' },
    { lat: 31.7774, lng: 76.9884, address: 'Upper Campus Road' },
    { lat: 31.7737, lng: 76.9844, address: 'Lower Campus Curve' },
    { lat: 31.7764, lng: 76.9874, address: 'New Hostel Road' },
];

const categories: HazardCategory[] = ["pothole", "damaged_pavement", "faded_markings", "roadside_obstruction", "drainage_issue", "poor_lighting", "driver_behavior", "other"];
const severities: Severity[] = ["low", "medium", "high", "critical"];
const statuses = ["submitted", "under_review", "in_progress", "resolved"];

const getRandomElement = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const getRandomDate = (start: Date, end: Date) => new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));

const seedData = async () => {
    try {
        await connectDB();
        
        // Clear existing
        console.log('Clearing old data...');
        await HazardReport.deleteMany({});
        await User.deleteMany({});
        
        // Create users
        console.log('Creating users...');
        const adminSalt = await bcrypt.genSalt(10);
        const adminHashedPassword = await bcrypt.hash('admin123', adminSalt);
        const admin = await User.create({
            name: 'Admin User',
            email: 'admin@roadsentinel.com',
            password: adminHashedPassword,
            role: 'admin'
        });

        const userSalt = await bcrypt.genSalt(10);
        const userHashedPassword = await bcrypt.hash('reporter123', userSalt);
        const reporter = await User.create({
            name: 'Campus Reporter',
            email: 'reporter@iitmandi.ac.in',
            password: userHashedPassword,
            role: 'user'
        });
        
        // Create reports
        console.log('Creating hazard reports...');
        const now = new Date();
        const sixMonthsAgo = new Date();
        sixMonthsAgo.setMonth(now.getMonth() - 6);

        const reportsToInsert = [];

        for (let i = 0; i < 30; i++) {
            const category = getRandomElement(categories);
            const severity = getRandomElement(severities);
            const status = getRandomElement(statuses);
            const location = campusLocations[i];
            const reportedAt = getRandomDate(sixMonthsAgo, now);
            const reportedBy = i % 3 === 0 ? admin._id : reporter._id;
            
            const title = `${category.replace('_', ' ').toUpperCase()} at ${location.address}`;
            const description = `This is a generated report for a ${severity} severity ${category.replace('_', ' ')}. Please look into it.`;
            
            const priorityScore = calculatePriorityScore(severity, category, reportedAt);
            
            reportsToInsert.push({
                title,
                description,
                category,
                severity,
                status,
                location,
                images: [],
                reportedBy,
                reportedAt,
                updatedAt: new Date(reportedAt.getTime() + 1000 * 60 * 60 * 24), // Updated 1 day later
                priorityScore
            });
        }
        
        await HazardReport.insertMany(reportsToInsert);
        
        console.log('Data seeded successfully!');
        console.log(`Created 2 users and ${reportsToInsert.length} hazard reports.`);
        
        process.exit();
    } catch (error: any) {
        console.error(`Seeding error: ${error.message}`);
        process.exit(1);
    }
};

seedData();
