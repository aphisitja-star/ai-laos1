"""
Sidebar — แถบด้านข้างสำหรับการตั้งค่าและควบคุมการวิเคราะห์
"""
import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """แสดง sidebar พร้อมการตั้งค่าทั้งหมด

    Returns:
        dict ของค่าการตั้งค่าที่ผู้ใช้เลือก
    """
    settings = {}

    with st.sidebar:
        st.markdown("## ⚙️ การตั้งค่าวิเคราะห์")
        st.markdown("---")

        # === Analysis Mode ===
        st.markdown("### 🔬 โหมดวิเคราะห์")
        settings["analysis_mode"] = st.selectbox(
            "เลือกโหมดการวิเคราะห์",
            options=["OpenCV (ไม่ต้องใช้โมเดล)", "CNN Model (ต้องเทรนโมเดล)"],
            index=0,
            help="OpenCV: ใช้การประมวลผลภาพ ทำงานได้ทันที\n"
            "CNN: ใช้โมเดล Deep Learning ต้องเทรนก่อน",
        )

        st.markdown("---")

        # === Cell Detection Settings ===
        st.markdown("### 🧬 การตรวจจับเซลล์")

        settings["min_cell_area"] = st.slider(
            "พื้นที่เซลล์ขั้นต่ำ (pixels)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="เซลล์ที่มีพื้นที่น้อยกว่านี้จะถูกข้ามไป",
        )

        settings["max_cell_area"] = st.slider(
            "พื้นที่เซลล์สูงสุด (pixels)",
            min_value=5000,
            max_value=50000,
            value=15000,
            step=1000,
            help="เซลล์ที่มีพื้นที่มากกว่านี้จะถูกข้ามไป",
        )

        settings["circularity_threshold"] = st.slider(
            "ค่า Circularity ขั้นต่ำ",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="ค่าความกลมของเซลล์ (0=ไม่กลม, 1=วงกลมสมบูรณ์)",
        )

        st.markdown("---")

        # === Image Processing Settings ===
        st.markdown("### 🖼️ การประมวลผลภาพ")

        settings["enhance_contrast"] = st.checkbox(
            "ปรับปรุงคอนทราสต์ (CLAHE)",
            value=True,
            help="ปรับปรุงคอนทราสต์ของภาพก่อนวิเคราะห์",
        )

        settings["noise_reduction"] = st.checkbox(
            "ลดสัญญาณรบกวน",
            value=True,
            help="ลด noise ในภาพด้วย Gaussian blur",
        )

        settings["blur_kernel"] = st.slider(
            "ขนาด Blur kernel",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            help="ขนาดของ kernel สำหรับ blur (ค่ามาก = blur มาก)",
        )

        st.markdown("---")

        # === Display Settings ===
        st.markdown("### 📊 การแสดงผล")

        settings["show_bounding_boxes"] = st.checkbox(
            "แสดงกรอบรอบเซลล์",
            value=True,
        )

        settings["show_cell_ids"] = st.checkbox(
            "แสดงหมายเลขเซลล์",
            value=True,
        )

        settings["show_heatmap"] = st.checkbox(
            "แสดง Heatmap การติดเชื้อ",
            value=True,
        )

        settings["show_cell_gallery"] = st.checkbox(
            "แสดงแกลเลอรีเซลล์",
            value=True,
        )

        st.markdown("---")

        # === About Section ===
        with st.expander("ℹ️ เกี่ยวกับแอปพลิเคชัน"):
            st.markdown(
                """
                **🔬 Blood Cell Microscope Analyzer**
                
                แอปพลิเคชันวิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์
                
                **ความสามารถ:**
                - 🧬 ตรวจจับเซลล์เม็ดเลือดแดง (RBC) และขาว (WBC)
                - 🦠 ตรวจหาเชื้อมาลาเรีย
                - 📊 คำนวณอัตราการติดเชื้อ (Parasitemia)
                - 📈 แสดงผลด้วยกราฟและสถิติ
                
                **ข้อจำกัด:**
                - เป็นเครื่องมือช่วยวิเคราะห์เบื้องต้น
                - ไม่สามารถทดแทนการวินิจฉัยของแพทย์
                - ผลลัพธ์ขึ้นกับคุณภาพภาพที่อัปโหลด
                
                **เวอร์ชัน:** 1.0  
                """
            )

        with st.expander("📖 วิธีใช้งาน"):
            st.markdown(
                """
                1. **อัปโหลดภาพ** — เลือกภาพเลือดจากกล้องจุลทรรศน์
                2. **ปรับการตั้งค่า** — กำหนดค่าการตรวจจับตามต้องการ
                3. **กดวิเคราะห์** — ระบบจะประมวลผลและแสดงผลลัพธ์
                4. **ดูผลลัพธ์** — ตรวจสอบภาพ สถิติ และกราฟ
                """
            )

    return settings
