"""
helpers.py — ฟังก์ชันอรรถประโยชน์สำหรับ Blood Cell Analyzer

ฟังก์ชันทั่วไปที่ใช้ร่วมกันทั่วทั้งแอปพลิเคชัน เช่น
โหลดภาพ, ปรับขนาด, จัดรูปแบบตัวเลข, ประเมินระดับความรุนแรง ฯลฯ
"""

from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import (
    CELL_COLORS,
    PARASITEMIA_LOW_THRESHOLD,
    PARASITEMIA_MODERATE_THRESHOLD,
    SEVERITY_LEVELS,
)


# ──────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────

def load_image(uploaded_file: UploadedFile) -> np.ndarray:
    """โหลดภาพจาก Streamlit UploadedFile แล้วแปลงเป็น BGR numpy array.

    Parameters
    ----------
    uploaded_file : UploadedFile
        ไฟล์ที่อัปโหลดผ่าน ``st.file_uploader``.

    Returns
    -------
    np.ndarray
        ภาพในรูปแบบ BGR (OpenCV convention), dtype uint8.

    Raises
    ------
    ValueError
        หากไม่สามารถถอดรหัสไฟล์ภาพได้.
    """
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"ไม่สามารถถอดรหัสไฟล์ภาพได้: {uploaded_file.name}"
        )
    return image


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """ปรับขนาดภาพโดยคงอัตราส่วนภาพเดิม.

    ด้านที่ยาวที่สุดจะถูกปรับให้ไม่เกิน *max_size* พิกเซล.
    หากภาพเล็กกว่า *max_size* อยู่แล้ว จะคืนภาพเดิมโดยไม่เปลี่ยนแปลง.

    Parameters
    ----------
    image : np.ndarray
        ภาพต้นฉบับ (BGR หรือ RGB).
    max_size : int, optional
        ขนาดพิกเซลสูงสุดของด้านที่ยาวที่สุด, default ``1024``.

    Returns
    -------
    np.ndarray
        ภาพที่ปรับขนาดแล้ว (หรือภาพเดิมถ้าไม่จำเป็นต้องปรับ).
    """
    h, w = image.shape[:2]
    longest_side = max(h, w)
    if longest_side <= max_size:
        return image

    scale = max_size / longest_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """แปลง numpy image เป็น base64-encoded string.

    Parameters
    ----------
    image : np.ndarray
        ภาพ BGR (OpenCV convention).
    fmt : str, optional
        รูปแบบการเข้ารหัส เช่น ``".png"`` หรือ ``".jpg"``, default ``".png"``.

    Returns
    -------
    str
        สตริง base64 ของภาพ (ไม่มี data-URI prefix).
    """
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError(f"ไม่สามารถเข้ารหัสภาพเป็นรูปแบบ {fmt} ได้")
    return base64.b64encode(buffer).decode("utf-8")


# ──────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────

def format_number(n: int | float) -> str:
    """จัดรูปแบบตัวเลขโดยใส่เครื่องหมายจุลภาค (comma).

    Parameters
    ----------
    n : int | float
        ตัวเลขที่ต้องการจัดรูปแบบ.

    Returns
    -------
    str
        สตริงตัวเลขที่มีเครื่องหมายจุลภาค เช่น ``"1,234"`` หรือ ``"1,234.56"``.
    """
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


# ──────────────────────────────────────────────
# Severity / Parasitemia
# ──────────────────────────────────────────────

def get_severity_label(parasitemia_rate: float) -> tuple[str, str]:
    """ประเมินระดับความรุนแรงจากอัตราการติดเชื้อ.

    Parameters
    ----------
    parasitemia_rate : float
        เปอร์เซ็นต์เซลล์ที่ติดเชื้อ (0-100).

    Returns
    -------
    tuple[str, str]
        ``(label, color)`` — ป้ายกำกับระดับความรุนแรง (ภาษาไทย)
        และรหัสสี hex.

    Examples
    --------
    >>> get_severity_label(0.5)
    ('ต่ำ', '#28a745')
    >>> get_severity_label(3.0)
    ('ปานกลาง', '#ffc107')
    >>> get_severity_label(8.0)
    ('สูง', '#dc3545')
    """
    if parasitemia_rate < PARASITEMIA_LOW_THRESHOLD:
        level = SEVERITY_LEVELS["LOW"]
    elif parasitemia_rate < PARASITEMIA_MODERATE_THRESHOLD:
        level = SEVERITY_LEVELS["MODERATE"]
    else:
        level = SEVERITY_LEVELS["HIGH"]

    return level["label_th"], level["color"]


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def calculate_statistics(cells_data: list[dict[str, Any]]) -> dict[str, Any]:
    """คำนวณสถิติพื้นฐานจากข้อมูลเซลล์ที่ตรวจพบ.

    Parameters
    ----------
    cells_data : list[dict[str, Any]]
        รายการ dict ของเซลล์ โดยแต่ละ dict ควรมีคีย์ ``"area"``
        (float) และ ``"type"`` (str) เป็นอย่างน้อย.

    Returns
    -------
    dict[str, Any]
        Dictionary ประกอบด้วย:
        - ``total_cells``  : int — จำนวนเซลล์ทั้งหมด
        - ``type_counts``  : dict[str, int] — จำนวนแยกตามประเภท
        - ``area_mean``    : float — ค่าเฉลี่ยพื้นที่
        - ``area_std``     : float — ส่วนเบี่ยงเบนมาตรฐานพื้นที่
        - ``area_min``     : float — พื้นที่น้อยสุด
        - ``area_max``     : float — พื้นที่มากสุด
    """
    if not cells_data:
        return {
            "total_cells": 0,
            "type_counts": {},
            "area_mean": 0.0,
            "area_std": 0.0,
            "area_min": 0.0,
            "area_max": 0.0,
        }

    areas = np.array([cell.get("area", 0.0) for cell in cells_data])

    # นับจำนวนเซลล์แต่ละประเภท
    type_counts: dict[str, int] = {}
    for cell in cells_data:
        cell_type = cell.get("type", "Unknown")
        type_counts[cell_type] = type_counts.get(cell_type, 0) + 1

    return {
        "total_cells": len(cells_data),
        "type_counts": type_counts,
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
    }


# ──────────────────────────────────────────────
# Color Legend
# ──────────────────────────────────────────────

def create_color_legend() -> dict[str, str]:
    """สร้าง mapping สีสำหรับแสดงผลประเภทเซลล์.

    Returns
    -------
    dict[str, str]
        Mapping ``{cell_type: hex_color}`` ที่ดึงมาจาก
        :py:data:`config.CELL_COLORS`.
    """
    return dict(CELL_COLORS)
