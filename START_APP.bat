@echo off
chcp 65001 >nul 2>&1
title 🔬 Blood Cell Microscope Analyzer

echo.
echo  ============================================
echo   🔬 Blood Cell Microscope Analyzer
echo   วิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์
echo  ============================================
echo.

cd /d "d:\AI LAOS\blood-cell-analyzer"

:: ตรวจสอบ venv
if not exist "venv\Scripts\activate.bat" (
    echo [!] ไม่พบ Virtual Environment — กำลังสร้างใหม่...
    python -m venv venv
    echo [+] ติดตั้ง dependencies...
    venv\Scripts\pip.exe install -r requirements.txt
)

echo [*] กำลังเปิดแอปพลิเคชัน...
echo [*] เปิดเบราว์เซอร์ไปที่: http://localhost:8501
echo.
echo    กด Ctrl+C เพื่อหยุดแอป
echo.

:: เปิดเบราว์เซอร์อัตโนมัติ
start "" http://localhost:8501

:: รัน Streamlit
call venv\Scripts\activate.bat
streamlit run app.py --server.headless true --server.port 8501

pause
