import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export interface MLPredictionResult {
  rawClass: string;
  label: string;
  confidence: number;
  confidencePercentage: string;
  mappedCategory?: string;
  suggestedSeverity?: string;
  topPredictions?: Array<{
    rawClass: string;
    label: string;
    confidence: number;
    confidencePercentage: string;
  }>;
  isLowConfidence?: boolean;
}

const FASTAPI_PREDICT_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000/api/v1/predict';

/**
 * Predicts the road hazard category from an uploaded image (base64 data URL or path).
 * Tries FastAPI HTTP endpoint first; falls back to direct Python classifier via stdin if offline.
 */
export async function predictRoadHazard(imageInput: string): Promise<MLPredictionResult | null> {
  if (!imageInput || typeof imageInput !== 'string') {
    return null;
  }

  // 1. Try FastAPI inference server via native Node fetch
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(FASTAPI_PREDICT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageInput }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data: any = await response.json();
      if (data && data.success) {
        return {
          rawClass: data.rawClass,
          label: data.label,
          confidence: data.confidence,
          confidencePercentage: data.confidencePercentage,
          mappedCategory: data.mappedCategory,
          suggestedSeverity: data.suggestedSeverity,
          topPredictions: data.topPredictions,
          isLowConfidence: data.isLowConfidence,
        };
      }
    }
  } catch (httpErr) {
    // FastAPI server not reachable or timed out; proceed to Python stdin fallback
  }

  // 2. Fallback: Run classifier.py directly via Python stdin
  try {
    return await new Promise<MLPredictionResult | null>((resolve) => {
      // Find classifier.py candidate paths
      const candidates = [
        path.resolve(__dirname, '..', '..', 'classifier.py'),
        path.resolve(__dirname, '..', 'classifier.py'),
        path.resolve(process.cwd(), 'classifier.py'),
        path.resolve(process.cwd(), 'fastapi-backend', 'classifier.py'),
        path.resolve(process.cwd(), '..', 'fastapi-backend', 'classifier.py'),
        path.resolve('C:\\Users\\nk748\\OneDrive\\Desktop\\DP-PROJECT\\fastapi-backend\\classifier.py'),
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
            if (data.success) {
              return resolve({
                rawClass: data.rawClass,
                label: data.label,
                confidence: data.confidence,
                confidencePercentage: data.confidencePercentage,
                mappedCategory: data.mappedCategory,
                suggestedSeverity: data.suggestedSeverity,
                topPredictions: data.topPredictions,
                isLowConfidence: data.isLowConfidence,
              });
            }
          }
        } catch (e) {
          // parse error
        }
        resolve(null);
      });

      child.on('error', () => {
        resolve(null);
      });

      // Write image to child stdin and close
      child.stdin.write(imageInput);
      child.stdin.end();
    });
  } catch (fallbackErr) {
    console.warn('[ML Integration] Inference fallback error:', fallbackErr);
    return null;
  }
}
