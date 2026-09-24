"""
🔬 Blood Cell Microscope Analyzer — Multi-Page Application
แอปพลิเคชันวิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์
"""

import streamlit as st
import numpy as np
import cv2
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    APP_TITLE, APP_SUBTITLE, PAGE_CONFIG, MIN_CELL_AREA,
    MAX_CELL_AREA, CIRCULARITY_THRESHOLD, TRAINED_MODELS_DIR, VERSION,
)
from utils.helpers import load_image, resize_image
from utils.report_generator import generate_certificate_html
from analysis.cell_detector import CellDetector
from analysis.disease_classifier import DiseaseClassifier
from analysis.image_processor import ImageProcessor
from ui.components import render_header, render_footer, upload_section, info_box
from ui.sidebar import render_sidebar
from ui.results_display import display_analysis_results


# ============================================================
# Page Config & Session State
# ============================================================

def configure_page():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown("""
        <style>
        .stApp { max-width: 1400px; margin: 0 auto; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8f9fa, #fff);
            border: 1px solid #eee; border-radius: 10px; padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
        .page-card {
            background: white; border: 1px solid #eee; border-radius: 12px;
            padding: 20px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .info-card {
            background: linear-gradient(135deg, #667eea11, #764ba211);
            border-left: 4px solid #667eea; border-radius: 8px;
            padding: 15px 20px; margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        "analysis_complete": False, "cells": None, "report": None,
        "annotated_image": None, "heatmap_image": None, "cell_stats": None,
        "cell_images": [], "original_image": None, "uploaded_filename": "",
        "analysis_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Analysis Pipeline
# ============================================================

def run_analysis(image, settings):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state["original_image"] = image_rgb

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.markdown("**⏳ 1/4:** กำลังประมวลผลภาพ...")
    progress_bar.progress(10)
    processor = ImageProcessor(image)
    if settings.get("enhance_contrast", True):
        processor.enhance_contrast()
    if settings.get("noise_reduction", True):
        processor.reduce_noise(kernel_size=settings.get("blur_kernel", 5))
    processed_image = processor.get_result()

    status_text.markdown("**⏳ 2/4:** กำลังตรวจจับเซลล์...")
    progress_bar.progress(30)
    detector = CellDetector(
        min_area=settings.get("min_cell_area", MIN_CELL_AREA),
        max_area=settings.get("max_cell_area", MAX_CELL_AREA),
        circularity_threshold=settings.get("circularity_threshold", CIRCULARITY_THRESHOLD),
    )
    cells = detector.detect_cells(processed_image)
    annotated_rgb = cv2.cvtColor(detector.draw_detections(image, cells), cv2.COLOR_BGR2RGB)
    cell_stats = detector.get_cell_statistics(cells)

    cell_images, cells_data = [], []
    for cell in cells:
        if cell.cropped_image is not None:
            cell_images.append(cv2.cvtColor(cell.cropped_image, cv2.COLOR_BGR2RGB))
        cells_data.append({
            "cell_id": cell.cell_id, "cell_type": cell.cell_type,
            "area": cell.area, "perimeter": cell.perimeter,
            "circularity": cell.circularity,
            "center_x": cell.center[0], "center_y": cell.center[1],
        })
    cell_stats["cells_data"] = cells_data
    cell_stats["cell_areas"] = [c.area for c in cells]

    status_text.markdown("**⏳ 3/4:** กำลังวิเคราะห์เชื้อมาลาเรีย...")
    progress_bar.progress(60)
    model_path = None
    if settings.get("analysis_mode", "").startswith("CNN"):
        mp = os.path.join(TRAINED_MODELS_DIR, "parasite_detector.h5")
        if os.path.exists(mp):
            model_path = mp
    classifier = DiseaseClassifier(model_path=model_path)
    report = classifier.analyze(image, cells)

    status_text.markdown("**⏳ 4/4:** กำลังสร้าง Heatmap...")
    progress_bar.progress(85)
    heatmap_rgb = cv2.cvtColor(
        classifier.create_heatmap(image, report.cell_results, cells), cv2.COLOR_BGR2RGB
    )

    st.session_state.update({
        "cells": cells, "report": report, "annotated_image": annotated_rgb,
        "heatmap_image": heatmap_rgb, "cell_stats": cell_stats,
        "cell_images": cell_images, "analysis_complete": True,
    })

    # Save to history
    st.session_state["analysis_history"].append({
        "datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "filename": st.session_state.get("uploaded_filename", ""),
        "total_cells": report.total_cells,
        "infected": report.infected_cells,
        "rate": report.parasitemia_rate,
        "severity": report.severity_level,
    })

    progress_bar.progress(100)
    status_text.markdown("**✅ วิเคราะห์เสร็จสิ้น!**")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()


# ============================================================
# PAGE: หน้าหลัก — วิเคราะห์ภาพเลือด
# ============================================================

def page_home():
    render_header()
    settings = render_sidebar()
    uploaded_file = upload_section()

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if image is not None:
            st.session_state["uploaded_filename"] = uploaded_file.name
            image = resize_image(image, max_size=2048)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with st.expander("👁️ ดูตัวอย่างภาพ", expanded=True):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(image_rgb, caption=f"📷 {uploaded_file.name} ({image.shape[1]}x{image.shape[0]} px)", use_container_width=True)

            st.markdown("")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("🔬 เริ่มวิเคราะห์", type="primary", use_container_width=True):
                    st.session_state["analysis_complete"] = False
                    run_analysis(image, settings)

            if st.session_state["analysis_complete"]:
                st.markdown("---")
                display_analysis_results(
                    st.session_state["original_image"], st.session_state["annotated_image"],
                    st.session_state["heatmap_image"], st.session_state["cell_stats"],
                    st.session_state["report"], st.session_state["cell_images"], settings,
                )
                # Certificate download
                st.markdown("---")
                st.markdown("## 📄 ใบรับรองผลการวิเคราะห์")
                try:
                    html = generate_certificate_html(
                        st.session_state["original_image"], st.session_state["annotated_image"],
                        st.session_state["heatmap_image"], st.session_state["cell_stats"],
                        st.session_state["report"], st.session_state.get("uploaded_filename", ""),
                    )
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.download_button("📄 ดาวน์โหลดใบรับรอง (HTML)", html.encode("utf-8"),
                                           "certificate.html", "text/html", type="primary", use_container_width=True)
                    st.info("💡 เปิดไฟล์ HTML ในเบราว์เซอร์ → กดปุ่ม 'พิมพ์' → เลือก 'Save as PDF'")
                except Exception as e:
                    st.error(f"❌ ไม่สามารถสร้างใบรับรอง: {e}")
        else:
            st.error("❌ ไม่สามารถโหลดภาพได้")
    else:
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="page-card" style="text-align:center"><h3>📤 ขั้นตอนที่ 1</h3><p>อัปโหลดภาพเลือด<br>จากกล้องจุลทรรศน์</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="page-card" style="text-align:center"><h3>🔬 ขั้นตอนที่ 2</h3><p>ระบบวิเคราะห์<br>เซลล์และเชื้อโรค</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="page-card" style="text-align:center"><h3>📊 ขั้นตอนที่ 3</h3><p>ดูผลลัพธ์<br>กราฟและสถิติ</p></div>', unsafe_allow_html=True)
        st.markdown("")
        info_box("วิธีใช้งาน", "อัปโหลดภาพเลือดที่ถ่ายจากกล้องจุลทรรศน์ ระบบจะวิเคราะห์เซลล์เม็ดเลือด ตรวจหาเชื้อมาลาเรีย และแสดงผลด้วยกราฟและสถิติ", "info")
    render_footer()


# ============================================================
# PAGE: ความรู้เรื่องมาลาเรีย
# ============================================================

def page_malaria_info():
    st.markdown("# 🦠 ความรู้เรื่องโรคมาลาเรีย")
    st.markdown("---")

    st.markdown("## มาลาเรียคืออะไร?")
    st.markdown("""
    **มาลาเรีย** (Malaria) เป็นโรคติดเชื้อที่เกิดจากเชื้อปรสิต **Plasmodium** ซึ่งแพร่กระจายผ่านยุงก้นปล่อง (Anopheles)
    เป็นโรคที่พบมากในเขตร้อนและกึ่งเขตร้อน รวมถึงประเทศไทย โดยเฉพาะบริเวณชายแดน
    """)

    st.markdown("## 🔬 ชนิดของเชื้อมาลาเรีย")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="page-card">
            <h4>🔴 Plasmodium falciparum</h4>
            <p><strong>อันตรายที่สุด</strong> — ทำให้เกิดมาลาเรียชนิดรุนแรง อาจถึงแก่ชีวิต</p>
            <ul>
                <li>ระยะฟักตัว: 7-14 วัน</li>
                <li>ไข้ทุก 36-48 ชั่วโมง</li>
                <li>อาจเกิดภาวะแทรกซ้อนรุนแรง</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="page-card">
            <h4>🟡 Plasmodium malariae</h4>
            <p>มาลาเรียชนิดไข้จตุรภาค — อาการไม่รุนแรงมาก</p>
            <ul>
                <li>ระยะฟักตัว: 18-40 วัน</li>
                <li>ไข้ทุก 72 ชั่วโมง</li>
                <li>อาจอยู่ในร่างกายนานหลายปี</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="page-card">
            <h4>🟢 Plasmodium vivax</h4>
            <p><strong>พบบ่อยที่สุดในไทย</strong> — อาการไม่รุนแรงมาก แต่กลับมาเป็นซ้ำได้</p>
            <ul>
                <li>ระยะฟักตัว: 12-17 วัน</li>
                <li>ไข้ทุก 48 ชั่วโมง</li>
                <li>มี hypnozoite ในตับ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="page-card">
            <h4>🔵 Plasmodium ovale</h4>
            <p>คล้าย P. vivax — พบน้อยในไทย</p>
            <ul>
                <li>ระยะฟักตัว: 16-18 วัน</li>
                <li>ไข้ทุก 48 ชั่วโมง</li>
                <li>กลับมาเป็นซ้ำได้เช่นกัน</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🤒 อาการของโรคมาลาเรีย")
    st.markdown("""
    | ระยะ | อาการ | ระยะเวลา |
    |:---|:---|:---|
    | **ระยะหนาวสั่น** | หนาวสั่น ตัวเย็น ริมฝีปากเขียว | 15-60 นาที |
    | **ระยะไข้สูง** | ไข้สูง 40-41°C ปวดศีรษะ คลื่นไส้ | 2-6 ชั่วโมง |
    | **ระยะเหงื่อออก** | เหงื่อออกมาก ไข้ลด อ่อนเพลีย | 2-4 ชั่วโมง |
    """)

    st.markdown("## 📊 ระดับ Parasitemia")
    st.markdown("""
    | ระดับ | Parasitemia Rate | ความรุนแรง |
    |:---|:---|:---|
    | 🟢 **ต่ำ** | < 1% | อาการไม่รุนแรง |
    | 🟡 **ปานกลาง** | 1 - 5% | ควรรักษาทันที |
    | 🔴 **สูง** | > 5% | อาจอันตรายถึงชีวิต |
    | 🔴 **สูงมาก** | > 10% | ภาวะฉุกเฉิน! |
    """)

    st.markdown("## 🛡️ การป้องกัน")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="page-card" style="text-align:center">
            <h1>🦟</h1>
            <h4>กันยุง</h4>
            <p>ใช้ยาทากันยุง นอนในมุ้ง ใส่เสื้อแขนยาว</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="page-card" style="text-align:center">
            <h1>💊</h1>
            <h4>ยาป้องกัน</h4>
            <p>รับยาป้องกันก่อนเดินทางไปพื้นที่เสี่ยง</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="page-card" style="text-align:center">
            <h1>🏥</h1>
            <h4>ตรวจเลือด</h4>
            <p>ตรวจเลือดทันทีเมื่อมีไข้หลังไปพื้นที่เสี่ยง</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning("⚠️ หากมีอาการไข้หลังเดินทางไปพื้นที่ที่มีมาลาเรียระบาด ให้รีบพบแพทย์ทันที!")


# ============================================================
# PAGE: สารานุกรมเซลล์เม็ดเลือด
# ============================================================

def page_blood_cells():
    st.markdown("# 🧬 สารานุกรมเซลล์เม็ดเลือด")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🔴 เม็ดเลือดแดง", "⚪ เม็ดเลือดขาว", "🟣 เกล็ดเลือด", "🩸 พลาสมา"])

    with tab1:
        st.markdown("## 🔴 เซลล์เม็ดเลือดแดง (Red Blood Cells / RBC)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
            **เม็ดเลือดแดง** หรือ Erythrocytes เป็นเซลล์ที่พบมากที่สุดในเลือด 
            มีหน้าที่ขนส่งออกซิเจนไปเลี้ยงเนื้อเยื่อทั่วร่างกาย

            | คุณสมบัติ | ค่า |
            |:---|:---|
            | **ขนาด** | 6-8 μm (ไมโครเมตร) |
            | **รูปร่าง** | จานเว้าสองด้าน (Biconcave disc) |
            | **จำนวนปกติ** | 4.5-5.5 ล้าน/μL |
            | **อายุขัย** | ~120 วัน |
            | **มีนิวเคลียส** | ❌ ไม่มี |
            | **สีในภาพย้อม** | ชมพู-แดง (ย้อม Giemsa) |
            """)
        with c2:
            st.info("💡 **ในภาพจุลทรรศน์:** RBC จะเห็นเป็นวงกลมสีชมพูมีส่วนกลางจางกว่าขอบ (central pallor)")

        st.markdown("""
        ### ความผิดปกติที่พบบ่อย
        - **Microcytic** — เซลล์เล็กกว่าปกติ (โลหิตจางจากขาดธาตุเหล็ก)
        - **Macrocytic** — เซลล์ใหญ่กว่าปกติ (ขาดวิตามิน B12/Folate)
        - **Sickle cell** — รูปร่างเป็นเคียว (โรคเม็ดเลือดแดงรูปเคียว)
        - **Spherocyte** — ทรงกลม ไม่เว้า (Hereditary spherocytosis)
        """)

    with tab2:
        st.markdown("## ⚪ เซลล์เม็ดเลือดขาว (White Blood Cells / WBC)")
        st.markdown("""
        **เม็ดเลือดขาว** หรือ Leukocytes ทำหน้าที่ป้องกันร่างกายจากเชื้อโรค

        | คุณสมบัติ | ค่า |
        |:---|:---|
        | **ขนาด** | 10-15 μm (ใหญ่กว่า RBC) |
        | **จำนวนปกติ** | 4,000-11,000/μL |
        | **มีนิวเคลียส** | ✅ มี (สีม่วงเข้ม) |
        | **สีในภาพย้อม** | นิวเคลียสม่วง-น้ำเงินเข้ม |
        """)

        st.markdown("### ชนิดของเม็ดเลือดขาว")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="page-card">
                <h4>Neutrophil (50-70%)</h4>
                <p>ด่านแรกในการป้องกัน กิน bacteria</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="page-card">
                <h4>Basophil (0-1%)</h4>
                <p>เกี่ยวกับปฏิกิริยาภูมิแพ้</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="page-card">
                <h4>Lymphocyte (20-40%)</h4>
                <p>T-cell, B-cell ภูมิคุ้มกันเฉพาะ</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="page-card">
                <h4>Eosinophil (1-4%)</h4>
                <p>ต่อต้านปรสิตและภูมิแพ้</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="page-card">
                <h4>Monocyte (2-8%)</h4>
                <p>กลายเป็น Macrophage กินเชื้อโรค</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("## 🟣 เกล็ดเลือด (Platelets / Thrombocytes)")
        st.markdown("""
        **เกล็ดเลือด** เป็นชิ้นส่วนของเซลล์ที่ช่วยในกระบวนการแข็งตัวของเลือด

        | คุณสมบัติ | ค่า |
        |:---|:---|
        | **ขนาด** | 2-3 μm (เล็กมาก) |
        | **จำนวนปกติ** | 150,000-400,000/μL |
        | **อายุขัย** | 8-10 วัน |
        | **หน้าที่** | ห้ามเลือด, สร้างลิ่มเลือด |
        """)

    with tab4:
        st.markdown("## 🩸 พลาสมา (Plasma)")
        st.markdown("""
        **พลาสมา** เป็นส่วนที่เป็นของเหลวของเลือด คิดเป็น ~55% ของปริมาตรเลือดทั้งหมด

        | ส่วนประกอบ | หน้าที่ |
        |:---|:---|
        | **น้ำ (90%)** | ตัวทำละลาย |
        | **โปรตีน (7%)** | Albumin, Globulin, Fibrinogen |
        | **เกลือแร่ (1%)** | Na⁺, K⁺, Ca²⁺ |
        | **สารอื่นๆ (2%)** | กลูโคส, ฮอร์โมน, วิตามิน |
        """)


# ============================================================
# PAGE: คู่มือกล้องจุลทรรศน์
# ============================================================

def page_microscope_guide():
    st.markdown("# 🔬 คู่มือการใช้กล้องจุลทรรศน์สำหรับตรวจเลือด")
    st.markdown("---")

    st.markdown("## 📋 ขั้นตอนการเตรียมสไลด์เลือด (Blood Smear)")

    steps = [
        ("1️⃣", "เก็บตัวอย่างเลือด", "เจาะเลือดจากปลายนิ้ว หยดเลือดลงบนสไลด์แก้ว"),
        ("2️⃣", "ทำ Thin Smear", "ใช้สไลด์อีกแผ่นเกลี่ยเลือดให้เป็นฟิล์มบาง ทำมุม 30-45°"),
        ("3️⃣", "ผึ่งให้แห้ง", "วางสไลด์ในที่อากาศถ่ายเทได้ ให้แห้งสนิท"),
        ("4️⃣", "ตรึงสไลด์", "แช่ในเมทานอลบริสุทธิ์ 1-2 นาที เพื่อตรึงเซลล์"),
        ("5️⃣", "ย้อมสี Giemsa", "แช่ในสารละลาย Giemsa 10% นาน 10-15 นาที"),
        ("6️⃣", "ล้างและผึ่งแห้ง", "ล้างด้วยน้ำกลั่น buffer pH 7.2 แล้วผึ่งให้แห้ง"),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="info-card">
            <strong>{icon} {title}</strong><br>
            {desc}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🔎 การตั้งกล้องจุลทรรศน์")
    st.markdown("""
    | กำลังขยาย | ใช้ทำอะไร |
    |:---|:---|
    | **10x (Low power)** | สำรวจภาพรวมของสไลด์ หาบริเวณที่เซลล์กระจายตัวดี |
    | **40x (High power)** | ดูรายละเอียดเซลล์เม็ดเลือด |
    | **100x (Oil immersion)** | ตรวจหาเชื้อมาลาเรีย (ต้องใช้น้ำมัน immersion) |
    """)

    st.markdown("## 📸 เคล็ดลับการถ่ายภาพสำหรับวิเคราะห์")
    c1, c2 = st.columns(2)
    with c1:
        st.success("""
        ✅ **ควรทำ:**
        - ใช้กำลังขยาย 100x Oil immersion
        - ปรับแสงให้สม่ำเสมอ (Köhler illumination)
        - โฟกัสให้คมชัด
        - ถ่ายบริเวณที่เซลล์ไม่ซ้อนกันมาก
        - บันทึกเป็น PNG หรือ TIFF (ไม่บีบอัด)
        """)
    with c2:
        st.error("""
        ❌ **ไม่ควรทำ:**
        - ถ่ายภาพมืดหรือสว่างเกินไป
        - ถ่ายบริเวณที่เซลล์ซ้อนกันหนาแน่น
        - ใช้ zoom ดิจิทัล (ลดคุณภาพ)
        - บันทึกเป็น JPEG quality ต่ำ
        - ถ่ายตอนสไลด์ยังเปียก
        """)


# ============================================================
# PAGE: ประวัติการวิเคราะห์
# ============================================================

def page_history():
    st.markdown("# 📋 ประวัติการวิเคราะห์")
    st.markdown("---")

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.info("📭 ยังไม่มีประวัติการวิเคราะห์ — ไปที่หน้า 'วิเคราะห์ภาพเลือด' เพื่อเริ่มต้น")
        return

    import pandas as pd
    df = pd.DataFrame(history)
    df.columns = ["วันเวลา", "ไฟล์ภาพ", "เซลล์ทั้งหมด", "ติดเชื้อ", "อัตรา (%)", "ระดับ"]

    st.metric("จำนวนการวิเคราะห์ทั้งหมด", f"{len(history)} ครั้ง")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("🗑️ ล้างประวัติ", type="secondary"):
        st.session_state["analysis_history"] = []
        st.rerun()


# ============================================================
# PAGE: คำถามที่พบบ่อย
# ============================================================

def page_faq():
    st.markdown("# ❓ คำถามที่พบบ่อย (FAQ)")
    st.markdown("---")

    faqs = [
        ("แอปนี้ใช้ตรวจโรคอะไรได้บ้าง?",
         "ปัจจุบันแอปรองรับการตรวจหาเชื้อ **มาลาเรีย** (Malaria) จากภาพเลือดย้อมสี Giemsa "
         "และสามารถนับแยกเซลล์เม็ดเลือดแดง (RBC) กับเม็ดเลือดขาว (WBC) ได้"),

        ("ภาพที่อัปโหลดต้องเป็นแบบไหน?",
         "ควรเป็นภาพถ่ายจากกล้องจุลทรรศน์กำลังขยาย 100x Oil immersion ของ thin blood smear "
         "ที่ย้อมสี Giemsa รองรับไฟล์ JPG, PNG, TIFF, BMP ขนาดไม่เกิน 200MB"),

        ("ผลวิเคราะห์แม่นยำแค่ไหน?",
         "โหมด OpenCV ให้ผลการตรวจคัดกรองเบื้องต้น ความแม่นยำขึ้นกับคุณภาพภาพ "
         "**ไม่สามารถใช้ทดแทนการวินิจฉัยของแพทย์ได้** ควรใช้เป็นเครื่องมือช่วยเท่านั้น"),

        ("CNN Model ดีกว่า OpenCV อย่างไร?",
         "CNN Model (ถ้าเทรนแล้ว) ใช้ Deep Learning วิเคราะห์ภาพ ให้ความแม่นยำสูงกว่า ~95% "
         "ในขณะที่ OpenCV ใช้การวิเคราะห์สีและรูปร่าง ทำงานได้ทันทีโดยไม่ต้องเทรน"),

        ("Parasitemia Rate คืออะไร?",
         "คือเปอร์เซ็นต์ของเซลล์เม็ดเลือดแดงที่ติดเชื้อมาลาเรีย เทียบกับเซลล์ทั้งหมด "
         "< 1% = ต่ำ, 1-5% = ปานกลาง, > 5% = สูง (อาจอันตราย)"),

        ("ใบรับรองผลวิเคราะห์ใช้อย่างไร?",
         "ดาวน์โหลดเป็นไฟล์ HTML เปิดในเบราว์เซอร์แล้วกด Print → Save as PDF "
         "ใช้เป็นเอกสารอ้างอิงเบื้องต้น แต่ **ไม่ใช่เอกสารทางการแพทย์**"),

        ("ปรับค่าอะไรได้บ้างที่ Sidebar?",
         "ปรับได้หลายค่า เช่น ขนาดเซลล์ min/max, ค่า Circularity, "
         "เปิด/ปิดการปรับคอนทราสต์, เปิด/ปิด Heatmap และแกลเลอรีเซลล์"),

        ("ทำไมตรวจไม่เจอเซลล์เลย?",
         "อาจเกิดจาก: 1) ภาพไม่ชัด/มืดเกินไป 2) กำลังขยายต่ำเกินไป 3) สไลด์ย้อมสีไม่ดี "
         "ลองปรับค่า 'พื้นที่เซลล์ขั้นต่ำ' ลดลง หรือเปิด 'ปรับปรุงคอนทราสต์'"),
    ]

    for q, a in faqs:
        with st.expander(f"💬 {q}"):
            st.markdown(a)


# ============================================================
# PAGE: เกี่ยวกับแอป
# ============================================================

def page_about():
    st.markdown("# ℹ️ เกี่ยวกับแอปพลิเคชัน")
    st.markdown("---")

    st.markdown(f"""
    <div class="page-card" style="text-align:center">
        <h1>🔬</h1>
        <h2>Blood Cell Microscope Analyzer</h2>
        <p style="font-size:1.1rem">วิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์ด้วย AI</p>
        <p><strong>Version {VERSION}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🛠️ เทคโนโลยีที่ใช้")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="page-card" style="text-align:center"><h3>🐍</h3><h4>Python</h4><p>ภาษาหลัก</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="page-card" style="text-align:center"><h3>👁️</h3><h4>OpenCV</h4><p>ประมวลผลภาพ</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="page-card" style="text-align:center"><h3>🧠</h3><h4>TensorFlow</h4><p>Deep Learning</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="page-card" style="text-align:center"><h3>📊</h3><h4>Streamlit</h4><p>Web UI</p></div>', unsafe_allow_html=True)

    st.markdown("## 📋 ความสามารถ")
    st.markdown("""
    | ฟีเจอร์ | รายละเอียด |
    |:---|:---|
    | 📤 อัปโหลดภาพ | รองรับ JPG, PNG, TIFF, BMP สูงสุด 200MB |
    | 🧬 ตรวจจับเซลล์ | แยก RBC / WBC ด้วย Watershed + Contour |
    | 🦠 ตรวจมาลาเรีย | OpenCV color analysis + CNN model |
    | 📊 กราฟวิเคราะห์ | Pie chart, Bar chart, Histogram, Heatmap |
    | 📄 ใบรับรอง | ดาวน์โหลดผลวิเคราะห์เป็น HTML/PDF |
    | 🎛️ ปรับแต่งได้ | ค่าการตรวจจับ, การแสดงผล |
    """)

    st.markdown("## ⚠️ ข้อจำกัดความรับผิดชอบ")
    st.warning(
        "แอปพลิเคชันนี้เป็น **เครื่องมือช่วยวิเคราะห์เบื้องต้น** เท่านั้น "
        "ไม่สามารถใช้ทดแทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้ "
        "ผลลัพธ์ขึ้นกับคุณภาพของภาพที่อัปโหลด"
    )


# ============================================================
# PAGE: วิธีแปลผล
# ============================================================

def page_interpretation():
    st.markdown("# 📖 วิธีแปลผลการวิเคราะห์")
    st.markdown("---")

    st.markdown("## 🖼️ การอ่านภาพผลลัพธ์")

    st.markdown("""
    ### 1. ภาพตรวจจับเซลล์
    | สี | ความหมาย |
    |:---|:---|
    | 🟢 **วงกลมสีเขียว** | เซลล์เม็ดเลือดแดง (RBC) |
    | 🔵 **วงกลมสีน้ำเงิน** | เซลล์เม็ดเลือดขาว (WBC) |
    | 🟡 **วงกลมสีเหลือง** | เซลล์ที่ไม่สามารถจำแนกได้ |

    ### 2. Heatmap การติดเชื้อ
    | สี | ความหมาย |
    |:---|:---|
    | 🔴 **แดง** | เซลล์ที่ตรวจพบเชื้อมาลาเรีย |
    | 🟢 **เขียว** | เซลล์ปกติ (ไม่ติดเชื้อ) |
    | ความเข้มของสี | สัมพันธ์กับ Confidence Score |
    """)

    st.markdown("## 📊 การอ่านกราฟ")
    st.markdown("""
    ### กราฟวงกลม (Pie Chart)
    แสดงสัดส่วนเซลล์ที่ติดเชื้อ vs ปกติ — ถ้าส่วนแดงเยอะ = ติดเชื้อมาก

    ### กราฟแท่ง (Bar Chart)
    แสดงจำนวนเซลล์แต่ละประเภท — ปกติ RBC จะมากกว่า WBC มาก

    ### Histogram
    แสดงการกระจายขนาดเซลล์ — ถ้ามี 2 กลุ่มชัดเจน แสดงว่ามี RBC + WBC
    """)

    st.markdown("## 🦠 การอ่านค่า Parasitemia")
    st.markdown("""
    <div class="info-card">
        <strong>สูตรคำนวณ:</strong><br>
        Parasitemia Rate (%) = (จำนวนเซลล์ติดเชื้อ ÷ จำนวนเซลล์ทั้งหมด) × 100
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="page-card" style="border-left:4px solid #28a745">
            <h4>🟢 ต่ำ (< 1%)</h4>
            <p>อาการไม่รุนแรง<br>รักษาตามปกติ</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="page-card" style="border-left:4px solid #ffc107">
            <h4>🟡 ปานกลาง (1-5%)</h4>
            <p>ควรเริ่มรักษาทันที<br>ติดตามอาการใกล้ชิด</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="page-card" style="border-left:4px solid #dc3545">
            <h4>🔴 สูง (> 5%)</h4>
            <p>อาจอันตรายถึงชีวิต<br>ต้องรักษาฉุกเฉิน!</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: คู่มือการย้อมสี
# ============================================================

def page_staining():
    st.markdown("# 🎨 คู่มือการย้อมสีเลือด")
    st.markdown("---")

    st.markdown("## วิธีย้อมสีที่นิยมใช้")

    tab1, tab2, tab3 = st.tabs(["Giemsa Stain", "Wright Stain", "Field Stain"])

    with tab1:
        st.markdown("""
        ## 💜 Giemsa Stain
        เป็นวิธีย้อมสีที่ **แนะนำสำหรับแอปนี้** เพราะให้ผลดีที่สุดในการตรวจมาลาเรีย

        ### สารเคมีที่ใช้
        - Giemsa stain stock solution
        - Buffer solution pH 7.2
        - Methanol (สำหรับตรึงสไลด์)

        ### ขั้นตอน
        1. ตรึงสไลด์ด้วย Methanol 1-2 นาที
        2. เตรียม Giemsa working solution (1:10 หรือ 1:20)
        3. แช่สไลด์ใน Giemsa 10-15 นาที
        4. ล้างด้วย buffer pH 7.2
        5. ผึ่งให้แห้ง

        ### ผลการย้อม
        | โครงสร้าง | สีที่ปรากฏ |
        |:---|:---|
        | **นิวเคลียส** | ม่วงเข้ม - แดงเข้ม |
        | **ไซโตพลาสซึม** | ฟ้าอ่อน |
        | **เม็ดเลือดแดง** | ชมพู - ส้มอ่อน |
        | **เชื้อมาลาเรีย** | ม่วงเข้ม - น้ำเงิน |
        | **Chromatin** | แดงเข้ม |
        """)

    with tab2:
        st.markdown("""
        ## 💙 Wright Stain
        นิยมใช้ในการตรวจ CBC (Complete Blood Count)

        ### ขั้นตอน
        1. หยด Wright stain ลงบนสไลด์
        2. ทิ้งไว้ 1-3 นาที
        3. เติม buffer pH 6.8 ในปริมาณเท่ากัน
        4. ทิ้งไว้ 3-5 นาที
        5. ล้างด้วยน้ำกลั่น
        """)

    with tab3:
        st.markdown("""
        ## 🩵 Field Stain
        ย้อมเร็ว เหมาะสำหรับ thick blood smear ตรวจมาลาเรียภาคสนาม

        ### ข้อดี
        - ย้อมเร็ว (ไม่กี่วินาที)
        - ใช้ในภาคสนามได้
        - ไม่ต้องตรึงสไลด์ก่อน
        """)


# ============================================================
# PAGE: ติดต่อ & ช่วยเหลือ
# ============================================================

def page_contact():
    st.markdown("# 📞 ติดต่อ & ขอความช่วยเหลือ")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="page-card">
            <h3>🏥 หากสงสัยว่าติดเชื้อมาลาเรีย</h3>
            <p>ติดต่อ <strong>สายด่วนกรมควบคุมโรค</strong></p>
            <h2 style="color:#dc3545">📞 1422</h2>
            <p>ให้บริการ 24 ชั่วโมง</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="page-card">
            <h3>🌐 แหล่งข้อมูลที่เป็นประโยชน์</h3>
            <ul>
                <li><a href="https://ddc.moph.go.th" target="_blank">กรมควบคุมโรค</a></li>
                <li><a href="https://www.who.int/malaria" target="_blank">WHO - Malaria</a></li>
                <li><a href="https://malaria.ddc.moph.go.th" target="_blank">สำนักงานป้องกันควบคุมโรค - มาลาเรีย</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="page-card">
            <h3>💬 แจ้งปัญหาการใช้งาน</h3>
            <p>หากพบปัญหาหรือมีข้อเสนอแนะ สามารถแจ้งได้ที่:</p>
            <ul>
                <li>📧 Email: support@bloodcellanalyzer.local</li>
                <li>📱 Line: @bloodcell</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="page-card">
            <h3>📝 ส่งข้อเสนอแนะ</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("feedback_form"):
            name = st.text_input("ชื่อ-สกุล")
            email = st.text_input("อีเมล")
            message = st.text_area("ข้อเสนอแนะ / ปัญหาที่พบ")
            if st.form_submit_button("📨 ส่งข้อเสนอแนะ", use_container_width=True):
                if message:
                    st.success("✅ ขอบคุณสำหรับข้อเสนอแนะ!")
                else:
                    st.warning("กรุณากรอกข้อเสนอแนะ")


# ============================================================
# MAIN — Navigation
# ============================================================

def main():
    configure_page()
    initialize_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 📌 เมนูหลัก")
        page = st.radio(
            "เลือกหน้า:",
            [
                "🔬 วิเคราะห์ภาพเลือด",
                "🦠 ความรู้เรื่องมาลาเรีย",
                "🧬 สารานุกรมเซลล์เม็ดเลือด",
                "📖 วิธีแปลผลวิเคราะห์",
                "🔬 คู่มือกล้องจุลทรรศน์",
                "🎨 คู่มือการย้อมสีเลือด",
                "📋 ประวัติการวิเคราะห์",
                "❓ คำถามที่พบบ่อย",
                "ℹ️ เกี่ยวกับแอป",
                "📞 ติดต่อ & ช่วยเหลือ",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")

    # Route to pages
    pages = {
        "🔬 วิเคราะห์ภาพเลือด": page_home,
        "🦠 ความรู้เรื่องมาลาเรีย": page_malaria_info,
        "🧬 สารานุกรมเซลล์เม็ดเลือด": page_blood_cells,
        "📖 วิธีแปลผลวิเคราะห์": page_interpretation,
        "🔬 คู่มือกล้องจุลทรรศน์": page_microscope_guide,
        "🎨 คู่มือการย้อมสีเลือด": page_staining,
        "📋 ประวัติการวิเคราะห์": page_history,
        "❓ คำถามที่พบบ่อย": page_faq,
        "ℹ️ เกี่ยวกับแอป": page_about,
        "📞 ติดต่อ & ช่วยเหลือ": page_contact,
    }

    pages[page]()


if __name__ == "__main__":
    main()
