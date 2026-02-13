@echo off
REM ============================================================
REM Scheduled Pipeline Runner for Windows Task Scheduler
REM ============================================================
REM 
REM Setup di Windows Task Scheduler:
REM 1. Buka Task Scheduler
REM 2. Create Basic Task
REM 3. Trigger: Daily atau Weekly
REM 4. Action: Start a program
REM 5. Program: path\to\run_scheduled.bat
REM 6. Start in: path\to\autoupdate_pipeline
REM
REM ============================================================

cd /d "%~dp0\.."

echo ============================================================
echo EARTHQUAKE MODEL AUTO-UPDATE PIPELINE
echo Scheduled Run: %date% %time%
echo ============================================================

REM Activate virtual environment if exists
if exist "..\..\.venv\Scripts\activate.bat" (
    call "..\..\.venv\Scripts\activate.bat"
)

REM Run daily check
python scripts/daily_check.py --run --auto-deploy

REM Log result
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Pipeline completed successfully
) else (
    echo [INFO] Pipeline not triggered or completed with issues
)

echo ============================================================
echo Completed: %date% %time%
echo ============================================================
