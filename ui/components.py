"""
UI Components — คอมโพเนนต์ UI ที่ใช้ซ้ำได้สำหรับแอปวิเคราะห์เซลล์เม็ดเลือด
"""
import streamlit as st
import numpy as np
from typing import Optional, Tuple


def render_header():
    """แสดงส่วนหัวของแอปพลิเคชัน"""
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #1E88E5;">🔬 Blood Cell Microscope Analyzer</h1>
            <p style="font-size: 1.2rem; color: #666;">
                วิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์ | ตรวจหาเชื้อโรค | วิเคราะห์เซลล์
            </p>
            <hr style="border: 1px solid #eee;">
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """แสดงส่วนท้ายของแอปพลิเคชัน"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem 0;">
            <p>⚠️ <strong>ข้อจำกัดความรับผิดชอบ:</strong> แอปพลิเคชันนี้เป็นเครื่องมือช่วยวิเคราะห์เบื้องต้นเท่านั้น 
            ไม่สามารถใช้ทดแทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้</p>
            <p>🔬 Blood Cell Microscope Analyzer v1.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "#1E88E5",
    icon: str = "",
):
    """แสดง metric card แบบกำหนดสไตล์ได้

    Args:
        title: ชื่อ metric
        value: ค่าที่จะแสดง
        subtitle: คำอธิบายเพิ่มเติม
        color: สีหลัก (hex)
        icon: emoji icon
    """
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}05);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 0.5rem 0;
        ">
            <p style="margin:0; font-size:0.85rem; color:#888;">{icon} {title}</p>
            <p style="margin:0; font-size:1.8rem; font-weight:700; color:{color};">{value}</p>
            <p style="margin:0; font-size:0.8rem; color:#aaa;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(level: str, color: str):
    """แสดง badge ระดับความรุนแรง

    Args:
        level: ข้อความระดับ เช่น 'ต่ำ (Low)'
        color: สี hex
    """
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {color};
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1rem;
            text-align: center;
        ">
            🦠 ระดับการติดเชื้อ: {level}
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_comparison(
    original: np.ndarray,
    analyzed: np.ndarray,
    label_left: str = "ภาพต้นฉบับ",
    label_right: str = "ภาพวิเคราะห์",
):
    """แสดงภาพเปรียบเทียบ side-by-side

    Args:
        original: ภาพต้นฉบับ (numpy array, RGB)
        analyzed: ภาพที่วิเคราะห์แล้ว (numpy array, RGB)
        label_left: ข้อความใต้ภาพซ้าย
        label_right: ข้อความใต้ภาพขวา
    """
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=label_left, use_container_width=True)
    with col2:
        st.image(analyzed, caption=label_right, use_container_width=True)


def upload_section() -> Optional[object]:
    """แสดงส่วนอัปโหลดภาพ พร้อม UI ที่สวยงาม

    Returns:
        UploadedFile object หรือ None ถ้ายังไม่อัปโหลด
    """
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 2rem;
            border: 2px dashed #1E88E5;
            border-radius: 12px;
            background: #f8f9fa;
            margin: 1rem 0;
        ">
            <h3 style="color: #1E88E5;">📤 อัปโหลดภาพเลือดจากกล้องจุลทรรศน์</h3>
            <p style="color: #666;">รองรับไฟล์: JPG, PNG, TIFF, BMP (สูงสุด 200MB)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "เลือกภาพเลือดจากกล้องจุลทรรศน์",
        type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        help="อัปโหลดภาพเลือดที่ถ่ายจากกล้องจุลทรรศน์ เพื่อวิเคราะห์เซลล์และตรวจหาเชื้อโรค",
        label_visibility="collapsed",
    )

    return uploaded_file


def progress_section(current_step: str, progress: float):
    """แสดง progress bar พร้อมข้อความสถานะ

    Args:
        current_step: ขั้นตอนปัจจุบัน
        progress: ค่า 0.0 - 1.0
    """
    st.markdown(f"**⏳ กำลังวิเคราะห์:** {current_step}")
    st.progress(progress)


def info_box(title: str, content: str, box_type: str = "info"):
    """แสดงกล่องข้อมูลแบบกำหนดสไตล์

    Args:
        title: หัวข้อ
        content: เนื้อหา
        box_type: 'info', 'success', 'warning', 'error'
    """
    colors = {
        "info": ("#1E88E5", "#E3F2FD"),
        "success": ("#28a745", "#E8F5E9"),
        "warning": ("#ffc107", "#FFF8E1"),
        "error": ("#dc3545", "#FFEBEE"),
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }

    border_color, bg_color = colors.get(box_type, colors["info"])
    icon = icons.get(box_type, "ℹ️")

    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border-left: 4px solid {border_color};
            border-radius: 4px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <strong>{icon} {title}</strong><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def cell_gallery(
    cell_images: list,
    labels: list,
    colors: list,
    cols_per_row: int = 6,
):
    """แสดง gallery ของเซลล์ที่ตรวจจับได้

    Args:
        cell_images: รายการ numpy array ของภาพเซลล์
        labels: ป้ายกำกับแต่ละเซลล์
        colors: สีกรอบแต่ละเซลล์
        cols_per_row: จำนวนคอลัมน์ต่อแถว
    """
    if not cell_images:
        st.info("ไม่พบเซลล์ที่ตรวจจับได้")
        return

    st.markdown("### 🔎 เซลล์ที่ตรวจจับได้")

    total = len(cell_images)
    for row_start in range(0, total, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < total:
                with col:
                    st.image(
                        cell_images[idx],
                        caption=labels[idx],
                        use_container_width=True,
                    )
