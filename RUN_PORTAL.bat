@echo off
title Edufi Hub Portal
cd /d "A:\Edufi_project"
echo ===================================================
echo   Starting Edufi Machine Learning & Security Hub
echo ===================================================
echo Opening http://127.0.0.1:8080 in your browser...
start http://127.0.0.1:8080
echo.
python app.py
pause
