"""
Results Display — แสดงผลลัพธ์การวิเคราะห์เซลล์เม็ดเลือด
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional


def display_analysis_results(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    heatmap_image: Optional[np.ndarray],
    cell_stats: dict,
    report: object,
    cell_images: list,
    settings: dict,
):
    """แสดงผลลัพธ์การวิเคราะห์ทั้งหมด

    Args:
        original_image: ภาพต้นฉบับ (RGB)
        annotated_image: ภาพที่มี bounding box (RGB)
        heatmap_image: ภาพ heatmap (RGB) หรือ None
        cell_stats: สถิติเซลล์จาก CellDetector
        report: AnalysisReport จาก DiseaseClassifier
        cell_images: รายการภาพเซลล์ที่ crop แล้ว
        settings: การตั้งค่าจาก sidebar
    """
    # === Summary Section ===
    st.markdown("## 📋 สรุปผลการวิเคราะห์")
    _display_summary_metrics(cell_stats, report)

    st.markdown("---")

    # === Severity Badge ===
    if report is not None:
        _display_severity(report)
        st.markdown("")

    # === Image Comparison ===
    st.markdown("## 🖼️ ภาพวิเคราะห์")
    _display_images(original_image, annotated_image, heatmap_image, settings)

    st.markdown("---")

    # === Charts ===
    st.markdown("## 📊 กราฟวิเคราะห์")
    _display_charts(cell_stats, report)

    st.markdown("---")

    # === Cell Gallery ===
    if settings.get("show_cell_gallery", True) and cell_images:
        st.markdown("## 🔎 แกลเลอรีเซลล์ที่ตรวจจับได้")
        _display_cell_gallery(cell_images, report)

    # === Detailed Table ===
    st.markdown("## 📝 รายละเอียดเซลล์")
    _display_cell_table(cell_stats, report)


def _display_summary_metrics(cell_stats: dict, report: object):
    """แสดงสถิติสรุปแบบ metric cards"""
    col1, col2, col3, col4 = st.columns(4)

    total_cells = cell_stats.get("total_cells", 0)
    rbc_count = cell_stats.get("rbc_count", 0)
    wbc_count = cell_stats.get("wbc_count", 0)

    with col1:
        st.metric(
            label="🧬 เซลล์ทั้งหมด",
            value=f"{total_cells:,}",
            help="จำนวนเซลล์ที่ตรวจจับได้ทั้งหมด",
        )

    with col2:
        st.metric(
            label="🔴 เม็ดเลือดแดง (RBC)",
            value=f"{rbc_count:,}",
            help="จำนวนเซลล์เม็ดเลือดแดง",
        )

    with col3:
        st.metric(
            label="🔵 เม็ดเลือดขาว (WBC)",
            value=f"{wbc_count:,}",
            help="จำนวนเซลล์เม็ดเลือดขาว",
        )

    with col4:
        if report is not None:
            infected = getattr(report, "infected_cells", 0)
            rate = getattr(report, "parasitemia_rate", 0.0)
            st.metric(
                label="🦠 เซลล์ติดเชื้อ",
                value=f"{infected:,}",
                delta=f"{rate:.2f}% Parasitemia",
                delta_color="inverse",
                help="จำนวนเซลล์ที่ตรวจพบเชื้อมาลาเรีย",
            )
        else:
            st.metric(label="🦠 เซลล์ติดเชื้อ", value="N/A")


def _display_severity(report: object):
    """แสดงระดับความรุนแรง"""
    severity = getattr(report, "severity_level", "ไม่ทราบ")
    color = getattr(report, "severity_color", "#888")

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {color};
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1.1rem;
        ">
            🦠 ระดับการติดเชื้อ: {severity}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary text
    if hasattr(report, "summary_thai"):
        st.markdown(f"**📋 สรุป:** {report.summary_thai}")


def _display_images(
    original: np.ndarray,
    annotated: np.ndarray,
    heatmap: Optional[np.ndarray],
    settings: dict,
):
    """แสดงภาพเปรียบเทียบ"""
    if heatmap is not None and settings.get("show_heatmap", True):
        # 3 columns: original, annotated, heatmap
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="📷 ภาพต้นฉบับ", use_container_width=True)
        with col2:
            st.image(
                annotated,
                caption="🔍 ภาพตรวจจับเซลล์",
                use_container_width=True,
            )
        with col3:
            st.image(
                heatmap,
                caption="🌡️ Heatmap การติดเชื้อ",
                use_container_width=True,
            )
    else:
        # 2 columns: original, annotated
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="📷 ภาพต้นฉบับ", use_container_width=True)
        with col2:
            st.image(
                annotated,
                caption="🔍 ภาพตรวจจับเซลล์",
                use_container_width=True,
            )


def _display_charts(cell_stats: dict, report: object):
    """แสดงกราฟวิเคราะห์"""
    col1, col2 = st.columns(2)

    with col1:
        _infection_pie_chart(report)

    with col2:
        _cell_type_bar_chart(cell_stats)

    # Cell size distribution
    if "cell_areas" in cell_stats and cell_stats["cell_areas"]:
        st.markdown("### 📏 การกระจายขนาดเซลล์")
        _cell_size_histogram(cell_stats["cell_areas"])


def _infection_pie_chart(report: object):
    """กราฟวงกลมแสดงสัดส่วนเซลล์ติดเชื้อ"""
    st.markdown("### 🦠 สัดส่วนการติดเชื้อ")

    if report is None:
        st.info("ไม่มีข้อมูลการวิเคราะห์เชื้อโรค")
        return

    infected = getattr(report, "infected_cells", 0)
    uninfected = getattr(report, "uninfected_cells", 0)

    if infected + uninfected == 0:
        st.info("ไม่พบเซลล์สำหรับวิเคราะห์")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["ติดเชื้อ (Infected)", "ปกติ (Uninfected)"],
                values=[infected, uninfected],
                marker=dict(colors=["#dc3545", "#28a745"]),
                hole=0.4,
                textinfo="label+percent+value",
                textfont_size=12,
            )
        ]
    )

    fig.update_layout(
        showlegend=True,
        height=350,
        margin=dict(t=30, b=30, l=30, r=30),
        font=dict(size=12),
    )

    st.plotly_chart(fig, use_container_width=True)


def _cell_type_bar_chart(cell_stats: dict):
    """กราฟแท่งแสดงจำนวนเซลล์แต่ละประเภท"""
    st.markdown("### 🧬 จำนวนเซลล์แต่ละประเภท")

    rbc = cell_stats.get("rbc_count", 0)
    wbc = cell_stats.get("wbc_count", 0)
    unknown = cell_stats.get("unknown_count", 0)

    categories = ["เม็ดเลือดแดง\n(RBC)", "เม็ดเลือดขาว\n(WBC)", "ไม่ระบุ\n(Unknown)"]
    values = [rbc, wbc, unknown]
    colors = ["#e74c3c", "#3498db", "#95a5a6"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        yaxis_title="จำนวน (เซลล์)",
        height=350,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _cell_size_histogram(areas: list):
    """ฮิสโตแกรมแสดงการกระจายขนาดเซลล์"""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=areas,
                nbinsx=30,
                marker_color="#1E88E5",
                opacity=0.75,
            )
        ]
    )

    fig.update_layout(
        xaxis_title="พื้นที่เซลล์ (pixels²)",
        yaxis_title="จำนวน (เซลล์)",
        height=300,
        margin=dict(t=20, b=40, l=50, r=20),
        showlegend=False,
        bargap=0.05,
    )

    st.plotly_chart(fig, use_container_width=True)


def _display_cell_gallery(cell_images: list, report: object):
    """แสดงแกลเลอรีเซลล์"""
    if not cell_images:
        st.info("ไม่มีภาพเซลล์ที่จะแสดง")
        return

    # Get infection results if available
    cell_results = []
    if report is not None and hasattr(report, "cell_results"):
        cell_results = report.cell_results

    # Create tabs for all / infected / uninfected
    tab_all, tab_infected, tab_normal = st.tabs(
        ["🧬 ทั้งหมด", "🦠 ติดเชื้อ", "✅ ปกติ"]
    )

    with tab_all:
        _render_cell_grid(cell_images, cell_results, filter_type="all")

    with tab_infected:
        _render_cell_grid(cell_images, cell_results, filter_type="infected")

    with tab_normal:
        _render_cell_grid(cell_images, cell_results, filter_type="normal")


def _render_cell_grid(
    cell_images: list,
    cell_results: list,
    filter_type: str = "all",
    cols_per_row: int = 8,
):
    """แสดง grid ของเซลล์"""
    display_items = []

    for i, img in enumerate(cell_images):
        is_infected = False
        confidence = 0.0

        if i < len(cell_results):
            result = cell_results[i]
            is_infected = getattr(result, "is_infected", False)
            confidence = getattr(result, "confidence", 0.0)

        if filter_type == "infected" and not is_infected:
            continue
        elif filter_type == "normal" and is_infected:
            continue

        label = f"#{i + 1}"
        if is_infected:
            label += f" 🦠 ({confidence:.0%})"
        else:
            label += " ✅"

        display_items.append((img, label))

    if not display_items:
        st.info(
            "ไม่พบเซลล์ในหมวดนี้"
            if filter_type != "all"
            else "ไม่พบเซลล์ที่ตรวจจับได้"
        )
        return

    st.caption(f"พบ {len(display_items)} เซลล์")

    for row_start in range(0, len(display_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(display_items):
                img, label = display_items[idx]
                with col:
                    st.image(img, caption=label, use_container_width=True)


def _display_cell_table(cell_stats: dict, report: object):
    """แสดงตารางรายละเอียดเซลล์"""
    import pandas as pd

    cells_data = cell_stats.get("cells_data", [])
    if not cells_data:
        st.info("ไม่มีข้อมูลรายละเอียดเซลล์")
        return

    cell_results = []
    if report is not None and hasattr(report, "cell_results"):
        cell_results = report.cell_results

    rows = []
    for cell in cells_data:
        cell_id = cell.get("cell_id", 0)
        row = {
            "เซลล์ #": cell_id,
            "ประเภท": cell.get("cell_type", "Unknown"),
            "พื้นที่ (px²)": f"{cell.get('area', 0):.0f}",
            "เส้นรอบวง (px)": f"{cell.get('perimeter', 0):.1f}",
            "ความกลม": f"{cell.get('circularity', 0):.2f}",
            "ตำแหน่ง (x,y)": f"({cell.get('center_x', 0)}, {cell.get('center_y', 0)})",
        }

        # Add infection status if available
        matching_result = None
        for r in cell_results:
            if getattr(r, "cell_id", -1) == cell_id:
                matching_result = r
                break

        if matching_result is not None:
            row["สถานะ"] = "🦠 ติดเชื้อ" if matching_result.is_infected else "✅ ปกติ"
            row["ความเชื่อมั่น"] = f"{matching_result.confidence:.1%}"
        else:
            row["สถานะ"] = "—"
            row["ความเชื่อมั่น"] = "—"

        rows.append(row)

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * len(rows) + 38),
    )
