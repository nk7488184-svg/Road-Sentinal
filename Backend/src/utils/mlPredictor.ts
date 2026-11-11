import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export interface MLPredictionItem {
  class: string;
  confidence: number;
  rawClass?: string;
  label?: string;
  confidencePercentage?: string;
  mappedCategory?: string;
  suggestedSeverity?: string;
}

export interface MLPredictionResult {
  filename?: string;
  prediction: string;
  confidence: number;
  low_confidence: boolean;
  top_predictions: MLPredictionItem[];
  // Enriched fields for backwards-compatibility
  success?: boolean;
  rawClass: string;
  label: string;
  confidencePercentage: string;
  mappedCategory: string;
  suggestedSeverity: string;
  topPredictions: MLPredictionItem[];
  isLowConfidence: boolean;
}

const ML_API_BASE_URL = (process.env.ML_API_URL || process.env.FASTAPI_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

const LABEL_FORMATTING: Record<string, string> = {
  "Broken_Staircase": "Broken Staircase / Curbside",
  "Debris_on_road": "Debris on Road",
  "Edge_Shoulder_Damage": "Edge & Shoulder Damage",
  "Iron_grid_damage": "Drainage Iron Grid Damage",
  "Longitudinal_Crack": "Longitudinal Road Crack",
  "Map(Block)_Cracking": "Map / Block Cracking",
  "Patch_Repair": "Patch Repair Deterioration",
  "Paver_Damage": "Paver / Brick Road Damage",
  "Pothole_Surface_Breakout": "Pothole & Surface Breakout",
  "Raveling_Loose_Aggregate": "Raveling & Loose Aggregate",
  "Rubble_Unpaved_surface": "Rubble & Unpaved Surface",
  "Transverse_Crack": "Transverse Road Crack",
  "Vegetation_Moss": "Vegetation & Moss Overgrowth",
};

const SEVERITY_MAPPING: Record<string, string> = {
  "Pothole_Surface_Breakout": "critical",
  "Broken_Staircase": "high",
  "Iron_grid_damage": "high",
  "Map(Block)_Cracking": "high",
  "Longitudinal_Crack": "medium",
  "Transverse_Crack": "medium",
  "Paver_Damage": "medium",
  "Edge_Shoulder_Damage": "medium",
  "Debris_on_road": "medium",
  "Rubble_Unpaved_surface": "medium",
  "Patch_Repair": "low",
  "Raveling_Loose_Aggregate": "low",
  "Vegetation_Moss": "low",
};

const CATEGORY_MAPPING: Record<string, string> = {
  "Broken_Staircase": "Broken_Staircase",
  "Debris_on_road": "Debris_on_road",
  "Edge_Shoulder_Damage": "Edge_Shoulder_Damage",
  "Iron_grid_damage": "Iron_grid_damage",
  "Longitudinal_Crack": "Longitudinal_Crack",
  "Map(Block)_Cracking": "Map(Block)_Cracking",
  "Patch_Repair": "Patch_Repair",
  "Paver_Damage": "Paver_Damage",
  "Pothole_Surface_Breakout": "Pothole_Surface_Breakout",
  "Raveling_Loose_Aggregate": "Raveling_Loose_Aggregate",
  "Rubble_Unpaved_surface": "Rubble_Unpaved_surface",
  "Transverse_Crack": "Transverse_Crack",
  "Vegetation_Moss": "Vegetation_Moss",
};

function formatPrediction(data: any): MLPredictionResult {
  const topClass = data.prediction || data.rawClass || (data.top_predictions?.[0]?.class) || 'Pothole_Surface_Breakout';
  const conf = typeof data.confidence === 'number' ? data.confidence : 0.0;
  const isLow = typeof data.low_confidence === 'boolean' ? data.low_confidence : (typeof data.isLowConfidence === 'boolean' ? data.isLowConfidence : conf < 0.60);

  const rawTopList = data.top_predictions || data.topPredictions || [];
  const topItems: MLPredictionItem[] = rawTopList.map((item: any) => {
    const cls = item.class || item.rawClass || topClass;
    const cVal = typeof item.confidence === 'number' ? item.confidence : conf;
    return {
      class: cls,
      rawClass: cls,
      label: LABEL_FORMATTING[cls] || cls.replace(/_/g, ' '),
      confidence: round(cVal, 4),
      confidencePercentage: `${round(cVal * 100, 1)}%`,
      mappedCategory: CATEGORY_MAPPING[cls] || 'other',
      suggestedSeverity: SEVERITY_MAPPING[cls] || 'medium',
    };
  });

  const bestItem = topItems[0] || {
    class: topClass,
    rawClass: topClass,
    label: LABEL_FORMATTING[topClass] || topClass.replace(/_/g, ' '),
    confidence: conf,
    confidencePercentage: `${round(conf * 100, 1)}%`,
    mappedCategory: CATEGORY_MAPPING[topClass] || 'other',
    suggestedSeverity: SEVERITY_MAPPING[topClass] || 'medium',
  };

  return {
    filename: data.filename || 'road.jpg',
    prediction: topClass,
    confidence: round(conf, 4),
    low_confidence: isLow,
    top_predictions: topItems.map(t => ({ class: t.class, confidence: t.confidence })),
    // Enriched properties
    success: true,
    rawClass: topClass,
    label: bestItem.label || LABEL_FORMATTING[topClass] || topClass,
    confidencePercentage: bestItem.confidencePercentage || `${round(conf * 100, 1)}%`,
    mappedCategory: bestItem.mappedCategory || CATEGORY_MAPPING[topClass] || 'other',
    suggestedSeverity: bestItem.suggestedSeverity || SEVERITY_MAPPING[topClass] || 'medium',
    topPredictions: topItems,
    isLowConfidence: isLow,
  };
}

function round(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

/**
 * Predicts the road hazard category from an uploaded image (base64 data URL, buffer or path).
 * Sends multipart/form-data to FastAPI /predict endpoint, falling back to direct Python classifier if offline.
 */
export async function predictRoadHazard(imageInput: string | Buffer): Promise<MLPredictionResult | null> {
  if (!imageInput) {
    return null;
  }

  // 1. Try FastAPI inference server using multipart/form-data
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const formData = new FormData();
    let fileBuffer: Buffer | null = null;
    let mimeType = 'image/jpeg';
    let filename = 'road.jpg';

    if (Buffer.isBuffer(imageInput)) {
      fileBuffer = imageInput;
    } else if (typeof imageInput === 'string') {
      const clean = imageInput.trim();
      if (clean.startsWith('data:image')) {
        const parts = clean.split(';base64,');
        if (parts.length === 2) {
          const mimeMatch = parts[0].match(/data:(.*?)$/);
          if (mimeMatch) mimeType = mimeMatch[1];
          fileBuffer = Buffer.from(parts[1], 'base64');
        }
      } else if (fs.existsSync(clean)) {
        fileBuffer = fs.readFileSync(clean);
        filename = path.basename(clean);
        if (filename.endsWith('.png')) mimeType = 'image/png';
        else if (filename.endsWith('.webp')) mimeType = 'image/webp';
      } else if (clean.length > 100) {
        // Raw base64 string
        try {
          fileBuffer = Buffer.from(clean, 'base64');
        } catch {
          // not base64
        }
      }
    }

    if (fileBuffer) {
      // Modern Node 18+ Blob & FormData
      const blob = new Blob([new Uint8Array(fileBuffer)], { type: mimeType });
      formData.append('file', blob, filename);

      const targetUrl = `${ML_API_BASE_URL}/predict`;
      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data: any = await response.json();
        if (data && (data.prediction || data.rawClass)) {
          return formatPrediction(data);
        }
      }
    }
  } catch (httpErr) {
    // ML server unreachable or timed out; proceed to Python fallback
  }

  // 2. Fallback: Run classifier.py directly via Python stdin
  try {
    return await new Promise<MLPredictionResult | null>((resolve) => {
      const candidates = [
        path.resolve(__dirname, '..', '..', 'classifier.py'),
        path.resolve(__dirname, '..', 'classifier.py'),
        path.resolve(process.cwd(), 'classifier.py'),
        path.resolve(process.cwd(), 'fastapi-backend', 'classifier.py'),
        path.resolve(process.cwd(), '..', 'fastapi-backend', 'classifier.py'),
        path.resolve('C:\\Users\\nk748\\OneDrive\\Desktop\\DP-PROJECT\\fastapi-backend\\classifier.py'),
        path.resolve('C:\\Users\\nk748\\OneDrive\\Desktop\\RD_API\\RD_API\\classifier.py'),
        path.resolve('C:\\Users\\nk748\\OneDrive\\Desktop\\Backend\\classifier.py'),
      ];

      const scriptPath = candidates.find(p => fs.existsSync(p)) || candidates[0];
      const child = spawn('python', [scriptPath, '--stdin']);
      let stdout = '';

      child.stdout.on('data', (d) => {
        stdout += d.toString();
      });

      child.on('close', (code) => {
        if (code !== 0 || !stdout) {
          return resolve(null);
        }
        try {
          const startIdx = stdout.indexOf('{');
          const endIdx = stdout.lastIndexOf('}');
          if (startIdx !== -1 && endIdx !== -1) {
            const jsonStr = stdout.substring(startIdx, endIdx + 1);
            const data = JSON.parse(jsonStr);
            if (data.prediction || data.rawClass) {
              return resolve(formatPrediction(data));
            }
          }
        } catch {
          // parse error
        }
        resolve(null);
      });

      child.on('error', () => {
        resolve(null);
      });

      const inputStr = typeof imageInput === 'string' ? imageInput : imageInput.toString('base64');
      child.stdin.write(inputStr);
      child.stdin.end();
    });
  } catch (fallbackErr) {
    console.warn('[ML Integration] Inference fallback error:', fallbackErr);
    return null;
  }
}
