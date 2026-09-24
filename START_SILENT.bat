@echo off
chcp 65001 >nul 2>&1

:: สคริปต์รันแบบเงียบ (ไม่มีหน้าต่าง cmd ค้าง) สำหรับ Startup
cd /d "d:\AI LAOS\blood-cell-analyzer"

if not exist "venv\Scripts\streamlit.exe" exit /b 1

start "" /min cmd /c "cd /d d:\AI LAOS\blood-cell-analyzer && call venv\Scripts\activate.bat && streamlit run app.py --server.headless true --server.port 8501"

:: รอ 3 วินาทีแล้วเปิดเบราว์เซอร์
timeout /t 3 /nobreak >nul
start "" http://localhost:8501
