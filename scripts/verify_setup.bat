@echo off
REM Quick verification script for Windows
REM Run this to check everything is ready before Docker build

echo ============================================================
echo SAM3D Setup Verification
echo ============================================================

echo.
echo [1/4] Checking checkpoints...
if exist "checkpoints\hf\pipeline.yaml" (
    echo   OK: pipeline.yaml found
) else (
    echo   ERROR: checkpoints not found!
    echo   Run: scripts\download_checkpoints.sh
    exit /b 1
)

if exist "checkpoints\hf\ss_generator.ckpt" (
    echo   OK: ss_generator.ckpt found (~6.4GB)
) else (
    echo   ERROR: ss_generator.ckpt not found!
    exit /b 1
)

echo.
echo [2/4] Checking required files...
if exist "Dockerfile" (echo   OK: Dockerfile) else (echo   MISSING: Dockerfile)
if exist "api_server.py" (echo   OK: api_server.py) else (echo   MISSING: api_server.py)
if exist "docker-compose.yml" (echo   OK: docker-compose.yml) else (echo   MISSING: docker-compose.yml)

echo.
echo [3/4] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   OK: Docker installed
    docker --version
) else (
    echo   WARNING: Docker not found. Install Docker Desktop.
)

echo.
echo [4/4] Checking scripts...
if exist "scripts\build.sh" (echo   OK: build.sh) else (echo   MISSING: build.sh)
if exist "scripts\push.sh" (echo   OK: push.sh) else (echo   MISSING: push.sh)

echo.
echo ============================================================
echo Verification complete!
echo.
echo Next steps:
echo   1. Build image:  bash scripts/build.sh
echo   2. Push image:   bash scripts/push.sh
echo   3. Deploy on RunPod
echo ============================================================

