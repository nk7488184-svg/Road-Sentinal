@echo off
echo ====================================================
echo Starting ROAD-SENTINEL ML Ecosystem
echo ====================================================

echo 1. Starting FastAPI ML Engine on port 8000 (pre-warmed EfficientNetV2-S)...
start "FastAPI ML Server (Port 8000)" cmd /k "python -m uvicorn main:app --app-dir fastapi-backend --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak > nul

echo 2. Starting Express Backend on port 3001...
start "Express Backend (Port 3001)" cmd /k "cd Backend && npm run dev"

timeout /t 2 /nobreak > nul

echo 3. Starting Public Website on port 3000...
start "Public Website (Port 3000)" cmd /k "cd ROAD-SENTINEL\campus-safety-website && npm run dev"

timeout /t 2 /nobreak > nul

echo 4. Starting Admin Dashboard on port 3002...
start "Admin Dashboard (Port 3002)" cmd /k "cd ROAD-SENTINEL\campus-safety-dashboard && npm run dev -- -p 3002"

echo ====================================================
echo All services launched!
echo - Public Website:  http://localhost:3000
echo - Admin Dashboard: http://localhost:3002
echo - Node Backend:    http://localhost:3001
echo - FastAPI Engine:  http://localhost:8000/docs
echo ====================================================
