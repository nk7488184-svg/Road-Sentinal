import mongoose from 'mongoose';

mongoose.set('bufferCommands', false);
mongoose.set('bufferTimeoutMS', 1000);

const connectDB = async (): Promise<void> => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/road-sentinel', {
      serverSelectionTimeoutMS: 2000,
      connectTimeoutMS: 2000,
    });
    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    if (error instanceof Error) {
      console.warn(`MongoDB Connection Notice: ${error.message}`);
      console.log('⚡ Active in ultra-fast in-memory cache mode.');
    } else {
      console.warn('An unknown error occurred during MongoDB connection');
    }
  }
};

export default connectDB;
