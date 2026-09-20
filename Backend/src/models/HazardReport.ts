import mongoose, { Document, Schema } from 'mongoose';

export type HazardCategory = 'pothole' | 'damaged_pavement' | 'faded_markings' | 'roadside_obstruction' | 'drainage_issue' | 'poor_lighting' | 'driver_behavior' | 'other';
export type Severity = 'low' | 'medium' | 'high' | 'critical';
export type Status = 'submitted' | 'under_review' | 'in_progress' | 'resolved';

export interface IHazardReport extends Document {
  title: string;
  description: string;
  category: HazardCategory;
  severity: Severity;
  status: Status;
  location: { lat: number; lng: number; address: string };
  images: string[];
  reportedBy: mongoose.Types.ObjectId | string;
  reportedAt: Date;
  updatedAt: Date;
  priorityScore: number;
}

const hazardReportSchema = new Schema<IHazardReport>(
  {
    title: {
      type: String,
      required: [true, 'Please add a title'],
      trim: true,
    },
    description: {
      type: String,
      required: [true, 'Please add a description'],
    },
    category: {
      type: String,
      enum: ['pothole', 'damaged_pavement', 'faded_markings', 'roadside_obstruction', 'drainage_issue', 'poor_lighting', 'driver_behavior', 'other'],
      required: [true, 'Please add a category'],
    },
    severity: {
      type: String,
      enum: ['low', 'medium', 'high', 'critical'],
      required: [true, 'Please add a severity'],
    },
    status: {
      type: String,
      enum: ['submitted', 'under_review', 'in_progress', 'resolved'],
      default: 'submitted',
    },
    location: {
      lat: {
        type: Number,
        required: [true, 'Latitude is required'],
      },
      lng: {
        type: Number,
        required: [true, 'Longitude is required'],
      },
      address: {
        type: String,
        required: [true, 'Address is required'],
      },
    },
    images: {
      type: [String],
      default: [],
    },
    reportedBy: {
      type: Schema.Types.Mixed, // Supports ObjectId or string for anonymous
      ref: 'User',
    },
    reportedAt: {
      type: Date,
      default: Date.now,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
    priorityScore: {
      type: Number,
      default: 0,
    },
  },
  {
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);

// Virtual for id
hazardReportSchema.virtual('id').get(function () {
  return this._id.toHexString();
});

// Update updatedAt on save
hazardReportSchema.pre<IHazardReport>('save', function (next) {
  this.updatedAt = new Date();
  next();
});

// Indexes
hazardReportSchema.index({ 'location.lat': 1, 'location.lng': 1 });
hazardReportSchema.index({ category: 1 });
hazardReportSchema.index({ status: 1 });
hazardReportSchema.index({ severity: 1 });

const HazardReport = mongoose.model<IHazardReport>('HazardReport', hazardReportSchema);

export default HazardReport;
