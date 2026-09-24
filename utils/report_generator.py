"""
report_generator.py — สร้างใบรับรองผลการวิเคราะห์เลือด (Certificate Report)

สร้างเอกสาร HTML ที่สามารถดาวน์โหลดและพิมพ์เป็น PDF ได้
"""
import base64
import io
from datetime import datetime

import cv2
import numpy as np


def _image_to_base64(image: np.ndarray, max_width: int = 400) -> str:
    """แปลงภาพ numpy array (RGB) เป็น base64 data URI สำหรับฝังใน HTML"""
    # Resize ถ้าใหญ่เกินไป
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    # Convert RGB to BGR for cv2.imencode
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    _, buffer = cv2.imencode(".png", bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_certificate_html(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    heatmap_image: np.ndarray,
    cell_stats: dict,
    report: object,
    filename: str = "uploaded_image",
) -> str:
    """สร้างใบรับรองผลการวิเคราะห์ในรูปแบบ HTML

    Args:
        original_image: ภาพต้นฉบับ (RGB)
        annotated_image: ภาพที่มี annotations (RGB)
        heatmap_image: ภาพ heatmap (RGB)
        cell_stats: สถิติเซลล์
        report: AnalysisReport
        filename: ชื่อไฟล์ที่อัปโหลด

    Returns:
        HTML string ของใบรับรอง
    """
    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M:%S")
    report_id = now.strftime("BCA-%Y%m%d-%H%M%S")

    # แปลงภาพเป็น base64
    orig_b64 = _image_to_base64(original_image, max_width=300)
    annot_b64 = _image_to_base64(annotated_image, max_width=300)
    heat_b64 = _image_to_base64(heatmap_image, max_width=300)

    # ดึงข้อมูลจาก report
    total_cells = getattr(report, "total_cells", 0)
    infected = getattr(report, "infected_cells", 0)
    uninfected = getattr(report, "uninfected_cells", 0)
    rate = getattr(report, "parasitemia_rate", 0.0)
    severity = getattr(report, "severity_level", "N/A")
    severity_color = getattr(report, "severity_color", "#888")
    method = getattr(report, "analysis_method", "opencv")

    rbc_count = cell_stats.get("rbc_count", 0)
    wbc_count = cell_stats.get("wbc_count", 0)
    avg_area = cell_stats.get("avg_area", 0)
    avg_circ = cell_stats.get("avg_circularity", 0)

    # สร้าง severity badge
    if rate < 1.0:
        severity_icon = "🟢"
        result_text = "ไม่พบการติดเชื้อที่มีนัยสำคัญ"
    elif rate < 5.0:
        severity_icon = "🟡"
        result_text = "พบการติดเชื้อระดับปานกลาง — ควรปรึกษาแพทย์"
    else:
        severity_icon = "🔴"
        result_text = "พบการติดเชื้อระดับสูง — ควรพบแพทย์ทันที"

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ใบรับรองผลวิเคราะห์ - {report_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Sarabun', 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}

        .certificate {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border: 3px solid #1E88E5;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}

        .header {{
            background: linear-gradient(135deg, #1E88E5, #1565C0);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 5px;
            letter-spacing: 2px;
        }}

        .header h2 {{
            font-size: 18px;
            font-weight: 400;
            opacity: 0.9;
        }}

        .header .report-id {{
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.8;
        }}

        .content {{
            padding: 30px;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding: 10px 0;
            margin-bottom: 5px;
        }}

        .info-row .label {{
            font-weight: 600;
            color: #555;
            min-width: 180px;
        }}

        .info-row .value {{
            text-align: right;
            font-weight: 400;
        }}

        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: #1E88E5;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #1E88E5;
        }}

        .severity-badge {{
            display: inline-block;
            background: {severity_color};
            color: white;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 16px;
            margin: 10px 0;
        }}

        .result-box {{
            background: #f8f9fa;
            border-left: 4px solid {severity_color};
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }}

        .result-box .result-text {{
            font-size: 16px;
            font-weight: 600;
        }}

        .images-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }}

        .image-card {{
            text-align: center;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 10px;
        }}

        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}

        .image-card .caption {{
            font-size: 13px;
            color: #666;
            margin-top: 8px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 12px;
            margin: 15px 0;
        }}

        .stat-card {{
            text-align: center;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px 10px;
            border: 1px solid #eee;
        }}

        .stat-card .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #1E88E5;
        }}

        .stat-card .stat-label {{
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }}

        .stat-card.danger .stat-value {{
            color: #dc3545;
        }}

        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 25px;
            font-size: 13px;
        }}

        .disclaimer strong {{
            color: #856404;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            border-top: 1px solid #eee;
        }}

        .footer .signatures {{
            display: flex;
            justify-content: space-around;
            margin-top: 40px;
            margin-bottom: 20px;
        }}

        .footer .sig-line {{
            width: 200px;
            text-align: center;
        }}

        .footer .sig-line .line {{
            border-top: 1px solid #333;
            margin-bottom: 5px;
        }}

        .footer .sig-line .label {{
            font-size: 13px;
            color: #666;
        }}

        .watermark {{
            text-align: center;
            font-size: 11px;
            color: #ccc;
            margin-top: 15px;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .certificate {{
                box-shadow: none;
                border: 2px solid #1E88E5;
            }}
            .no-print {{
                display: none !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="text-align:center; margin-bottom:15px;">
        <button onclick="window.print()" style="
            background:#1E88E5; color:white; border:none; padding:12px 30px;
            border-radius:8px; font-size:16px; cursor:pointer; font-family:inherit;
        ">🖨️ พิมพ์ใบรับรอง / บันทึกเป็น PDF</button>
    </div>

    <div class="certificate">
        <!-- Header -->
        <div class="header">
            <h1>🔬 ใบรับรองผลการวิเคราะห์เลือด</h1>
            <h2>Blood Cell Microscope Analysis Certificate</h2>
            <div class="report-id">รหัสรายงาน: {report_id}</div>
        </div>

        <div class="content">
            <!-- ข้อมูลทั่วไป -->
            <div class="section-title">📋 ข้อมูลทั่วไป</div>
            <div class="info-row">
                <span class="label">📅 วันที่วิเคราะห์:</span>
                <span class="value">{date_str}</span>
            </div>
            <div class="info-row">
                <span class="label">⏰ เวลา:</span>
                <span class="value">{time_str}</span>
            </div>
            <div class="info-row">
                <span class="label">📁 ไฟล์ภาพ:</span>
                <span class="value">{filename}</span>
            </div>
            <div class="info-row">
                <span class="label">🔬 วิธีการวิเคราะห์:</span>
                <span class="value">{method.upper()} Analysis</span>
            </div>

            <!-- ผลวิเคราะห์หลัก -->
            <div class="section-title">🦠 ผลการตรวจหาเชื้อมาลาเรีย</div>

            <div style="text-align:center;">
                <div class="severity-badge">{severity_icon} ระดับการติดเชื้อ: {severity}</div>
            </div>

            <div class="result-box">
                <div class="result-text">{result_text}</div>
                <p style="margin-top:8px; font-size:14px; color:#666;">
                    อัตราการติดเชื้อ (Parasitemia Rate): <strong>{rate:.2f}%</strong>
                </p>
            </div>

            <!-- สถิติ -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_cells:,}</div>
                    <div class="stat-label">เซลล์ทั้งหมด</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{rbc_count:,}</div>
                    <div class="stat-label">เม็ดเลือดแดง (RBC)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{wbc_count:,}</div>
                    <div class="stat-label">เม็ดเลือดขาว (WBC)</div>
                </div>
                <div class="stat-card danger">
                    <div class="stat-value">{infected:,}</div>
                    <div class="stat-label">เซลล์ติดเชื้อ</div>
                </div>
            </div>

            <!-- รายละเอียดเซลล์ -->
            <div class="section-title">🧬 รายละเอียดเซลล์</div>
            <div class="info-row">
                <span class="label">จำนวนเซลล์ทั้งหมด:</span>
                <span class="value">{total_cells:,} เซลล์</span>
            </div>
            <div class="info-row">
                <span class="label">เม็ดเลือดแดง (RBC):</span>
                <span class="value">{rbc_count:,} เซลล์</span>
            </div>
            <div class="info-row">
                <span class="label">เม็ดเลือดขาว (WBC):</span>
                <span class="value">{wbc_count:,} เซลล์</span>
            </div>
            <div class="info-row">
                <span class="label">เซลล์ที่ติดเชื้อ:</span>
                <span class="value" style="color:{severity_color}; font-weight:700;">{infected:,} เซลล์ ({rate:.2f}%)</span>
            </div>
            <div class="info-row">
                <span class="label">เซลล์ปกติ:</span>
                <span class="value" style="color:#28a745;">{uninfected:,} เซลล์</span>
            </div>
            <div class="info-row">
                <span class="label">ขนาดเซลล์เฉลี่ย:</span>
                <span class="value">{avg_area:.0f} px²</span>
            </div>
            <div class="info-row">
                <span class="label">ความกลมเฉลี่ย:</span>
                <span class="value">{avg_circ:.4f}</span>
            </div>

            <!-- ภาพวิเคราะห์ -->
            <div class="section-title">🖼️ ภาพประกอบการวิเคราะห์</div>
            <div class="images-grid">
                <div class="image-card">
                    <img src="{orig_b64}" alt="Original">
                    <div class="caption">📷 ภาพต้นฉบับ</div>
                </div>
                <div class="image-card">
                    <img src="{annot_b64}" alt="Annotated">
                    <div class="caption">🔍 ภาพตรวจจับเซลล์</div>
                </div>
                <div class="image-card">
                    <img src="{heat_b64}" alt="Heatmap">
                    <div class="caption">🌡️ Heatmap การติดเชื้อ</div>
                </div>
            </div>

            <!-- คำเตือน -->
            <div class="disclaimer">
                <strong>⚠️ ข้อจำกัดความรับผิดชอบ (Disclaimer):</strong><br>
                ผลการวิเคราะห์นี้เป็นเพียงการตรวจคัดกรองเบื้องต้นด้วยระบบคอมพิวเตอร์เท่านั้น 
                <strong>ไม่สามารถใช้ทดแทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้</strong>
                ผลลัพธ์อาจมีความคลาดเคลื่อนขึ้นกับคุณภาพของภาพ การย้อมสี และสภาพแวดล้อม 
                กรุณาปรึกษาแพทย์เพื่อรับการวินิจฉัยและรักษาที่ถูกต้อง
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <div class="signatures">
                <div class="sig-line">
                    <div class="line"></div>
                    <div class="label">ผู้ส่งตรวจ</div>
                </div>
                <div class="sig-line">
                    <div class="line"></div>
                    <div class="label">ผู้ตรวจสอบ</div>
                </div>
                <div class="sig-line">
                    <div class="line"></div>
                    <div class="label">ผู้รับรอง</div>
                </div>
            </div>
            <div class="watermark">
                🔬 Blood Cell Microscope Analyzer v1.0 | สร้างเมื่อ {date_str} {time_str}
            </div>
        </div>
    </div>
</body>
</html>"""

    return html
