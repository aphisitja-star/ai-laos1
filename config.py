"""
config.py — ไฟล์การตั้งค่าหลักสำหรับ Blood Cell Microscope Analyzer

รวมค่าคงที่ทั้งหมดที่ใช้ในแอปพลิเคชัน ได้แก่:
- ข้อมูลแอป (ชื่อ, คำบรรยาย)
- พารามิเตอร์การตรวจจับเซลล์
- ช่วงสี HSV สำหรับแยกประเภทเซลล์
- ระดับความรุนแรงของการติดเชื้อ
- เส้นทางโมเดลและการตั้งค่า Streamlit
"""

from typing import Final
import os
import numpy as np

# Project root directory
_PROJECT_ROOT: Final[str] = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR: Final[str] = os.path.join(_PROJECT_ROOT, "trained_models")

# ──────────────────────────────────────────────
# App Information / ข้อมูลแอปพลิเคชัน
# ──────────────────────────────────────────────
APP_TITLE: Final[str] = "🔬 Blood Cell Microscope Analyzer"
APP_SUBTITLE: Final[str] = "วิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์"
APP_VERSION: Final[str] = "1.0.0"
VERSION: Final[str] = APP_VERSION

# ──────────────────────────────────────────────
# File Upload Settings / การตั้งค่าอัปโหลดไฟล์
# ──────────────────────────────────────────────
SUPPORTED_FORMATS: Final[list[str]] = ["jpg", "jpeg", "png", "tiff", "tif", "bmp"]
MAX_FILE_SIZE_MB: Final[int] = 200

# ──────────────────────────────────────────────
# Model Input Settings / การตั้งค่าอินพุตโมเดล
# ──────────────────────────────────────────────
IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)

# ──────────────────────────────────────────────
# Cell Detection Parameters / พารามิเตอร์การตรวจจับเซลล์
# ──────────────────────────────────────────────
MIN_CELL_AREA: Final[int] = 500        # พื้นที่ขั้นต่ำ (pixels²) เพื่อกรอง noise
MAX_CELL_AREA: Final[int] = 15000      # พื้นที่สูงสุด (pixels²) เพื่อกรองวัตถุใหญ่เกิน
CIRCULARITY_THRESHOLD: Final[float] = 0.5  # ค่าความกลมขั้นต่ำ (0-1)

# ──────────────────────────────────────────────
# HSV Color Ranges / ช่วงสี HSV สำหรับตรวจจับเซลล์
# ──────────────────────────────────────────────
# Red Blood Cells (เม็ดเลือดแดง) — โทนแดง-ส้มในภาพย้อม
RBC_HSV_LOWER: Final[np.ndarray] = np.array([0, 50, 50])
RBC_HSV_UPPER: Final[np.ndarray] = np.array([20, 255, 255])
RBC_HSV_LOWER2: Final[np.ndarray] = np.array([160, 50, 50])
RBC_HSV_UPPER2: Final[np.ndarray] = np.array([180, 255, 255])

# White Blood Cells (เม็ดเลือดขาว) — นิวเคลียสสีม่วง-น้ำเงินเข้ม
WBC_HSV_LOWER: Final[np.ndarray] = np.array([120, 40, 40])
WBC_HSV_UPPER: Final[np.ndarray] = np.array([160, 255, 255])

# Platelet (เกล็ดเลือด) — จุดเล็กสีม่วงอ่อน
PLATELET_HSV_LOWER: Final[np.ndarray] = np.array([130, 30, 100])
PLATELET_HSV_UPPER: Final[np.ndarray] = np.array([170, 200, 255])

# ──────────────────────────────────────────────
# Parasitemia Severity Levels / ระดับความรุนแรง
# ──────────────────────────────────────────────
PARASITEMIA_LOW_THRESHOLD: Final[float] = 1.0       # < 1% = ต่ำ
PARASITEMIA_MODERATE_THRESHOLD: Final[float] = 5.0   # 1-5% = ปานกลาง
# > 5% = สูง

SEVERITY_LEVELS: Final[dict[str, dict[str, str]]] = {
    "LOW": {
        "label_th": "ต่ำ",
        "label_en": "Low",
        "color": "#28a745",       # เขียว
        "description": "การติดเชื้อระดับต่ำ (< 1%)",
    },
    "MODERATE": {
        "label_th": "ปานกลาง",
        "label_en": "Moderate",
        "color": "#ffc107",       # เหลือง
        "description": "การติดเชื้อระดับปานกลาง (1-5%)",
    },
    "HIGH": {
        "label_th": "สูง",
        "label_en": "High",
        "color": "#dc3545",       # แดง
        "description": "การติดเชื้อระดับสูง (> 5%)",
    },
}

# ──────────────────────────────────────────────
# Cell Type Display Settings / สีแสดงผลเซลล์
# ──────────────────────────────────────────────
CELL_COLORS: Final[dict[str, str]] = {
    "RBC": "#e74c3c",          # แดง
    "WBC": "#3498db",          # น้ำเงิน
    "Platelet": "#9b59b6",     # ม่วง
    "Infected": "#f39c12",     # ส้ม
    "Unknown": "#95a5a6",      # เทา
}

# ──────────────────────────────────────────────
# Model Paths / เส้นทางโมเดล
# ──────────────────────────────────────────────
MODEL_DIR: Final[str] = "models"
CELL_CLASSIFIER_PATH: Final[str] = f"{MODEL_DIR}/cell_classifier.h5"
PARASITE_DETECTOR_PATH: Final[str] = f"{MODEL_DIR}/parasite_detector.h5"

# ──────────────────────────────────────────────
# Streamlit Page Config / การตั้งค่าหน้าเว็บ
# ──────────────────────────────────────────────
PAGE_CONFIG: Final[dict] = {
    "page_title": "Blood Cell Analyzer",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
