@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts/get_criteo_sample.py --rows 150000
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts/train.py --data criteo
if errorlevel 1 goto failed
echo Completed. Restart the dashboard to select criteo.
pause
exit /b 0
:failed
echo Failed. Please keep this window open to inspect the error.
pause
exit /b 1
