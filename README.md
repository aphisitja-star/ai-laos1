# 🔬 Blood Cell Microscope Analyzer

แอปพลิเคชันวิเคราะห์ภาพเลือดจากกล้องจุลทรรศน์ เพื่อตรวจหาเชื้อโรค (มาลาเรีย) และวิเคราะห์เซลล์เม็ดเลือด

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## ✨ ความสามารถ

- 📤 **อัปโหลดภาพ** — รองรับ JPG, PNG, TIFF, BMP จากกล้องจุลทรรศน์
- 🧬 **ตรวจจับเซลล์** — ตรวจจับเซลล์เม็ดเลือดแดง (RBC) และเม็ดเลือดขาว (WBC)
- 🦠 **วิเคราะห์เชื้อโรค** — ตรวจหาเชื้อมาลาเรียในเซลล์เม็ดเลือดแดง
- 📊 **แสดงผลลัพธ์** — กราฟ, Heatmap, สถิติ, และตารางรายละเอียด
- 🎛️ **ปรับแต่งได้** — ปรับค่าการตรวจจับผ่าน sidebar

## 🚀 วิธีติดตั้ง

### 1. สร้าง Virtual Environment (แนะนำ)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 3. รันแอปพลิเคชัน

```bash
streamlit run app.py
```

แอปจะเปิดที่ `http://localhost:8501`

## 📖 วิธีใช้งาน

1. **อัปโหลดภาพ** — เลือกภาพเลือดจากกล้องจุลทรรศน์ (ย้อมสี Giemsa/Wright)
2. **ปรับการตั้งค่า** — กำหนดค่าการตรวจจับที่แถบด้านข้าง (sidebar)
3. **กดวิเคราะห์** — ระบบจะประมวลผลและแสดงผลลัพธ์
4. **ดูผลลัพธ์** — ตรวจสอบภาพ สถิติ กราฟ และ Heatmap

## 🔬 โหมดการวิเคราะห์

### OpenCV Mode (เริ่มต้น)
- ใช้การประมวลผลภาพด้วย OpenCV
- ไม่ต้องเทรนโมเดล — ทำงานได้ทันที
- วิเคราะห์สีและ morphology ของเซลล์

### CNN Model Mode (ขั้นสูง)
- ใช้ MobileNetV2 transfer learning
- ต้องเทรนโมเดลก่อนใช้งาน
- แม่นยำกว่า OpenCV mode

#### วิธีเทรนโมเดล

```bash
# ดาวน์โหลด dataset
python models/train_model.py --download

# เทรนโมเดล
python models/train_model.py --train

# หรือทำทั้งหมด
python models/train_model.py --all
```

## 📁 โครงสร้างโปรเจกต์

```
blood-cell-analyzer/
├── app.py                    # แอปพลิเคชันหลัก
├── config.py                 # การตั้งค่า
├── requirements.txt          # Dependencies
├── README.md                 # เอกสารนี้
├── analysis/                 # โมดูลวิเคราะห์
│   ├── image_processor.py    # ประมวลผลภาพ
│   ├── cell_detector.py      # ตรวจจับเซลล์
│   └── disease_classifier.py # จำแนกเชื้อโรค
├── models/                   # สคริปต์เทรนโมเดล
│   └── train_model.py
├── ui/                       # คอมโพเนนต์ UI
│   ├── components.py         # คอมโพเนนต์ทั่วไป
│   ├── sidebar.py            # แถบด้านข้าง
│   └── results_display.py    # แสดงผลลัพธ์
├── utils/                    # ฟังก์ชันช่วย
│   └── helpers.py
└── trained_models/           # โมเดลที่เทรนแล้ว
```

## ⚠️ ข้อจำกัดความรับผิดชอบ

แอปพลิเคชันนี้เป็น **เครื่องมือช่วยวิเคราะห์เบื้องต้น** เท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยของแพทย์ผู้เชี่ยวชาญได้ ผลลัพธ์ขึ้นกับคุณภาพของภาพที่อัปโหลด

## 📋 ข้อกำหนดระบบ

- Python 3.10 ขึ้นไป
- RAM: 4GB+ (แนะนำ 8GB สำหรับเทรนโมเดล)
- Storage: ~2GB (รวม dataset สำหรับเทรน)
