"""
cell_detector.py — โมดูลตรวจจับเซลล์เม็ดเลือดจากภาพกล้องจุลทรรศน์
Core cell detection module using OpenCV for blood cell microscope images.

Pipeline:
    1. Preprocessing (Gaussian blur + CLAHE)
    2. Adaptive thresholding
    3. Morphological cleanup
    4. Watershed segmentation for overlapping cells
    5. Contour extraction and filtering
    6. RBC / WBC classification by size and HSV colour
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Data class สำหรับเก็บข้อมูลเซลล์ที่ตรวจจับได้
# ---------------------------------------------------------------------------

@dataclass
class CellInfo:
    """Information about a single detected cell.

    Attributes:
        cell_id: Unique identifier for the cell within the image.
        center: (x, y) coordinate of the cell centre.
        radius: Approximate radius (pixels).
        area: Contour area (pixels²).
        perimeter: Contour perimeter (pixels).
        circularity: 4π·area / perimeter²  (1.0 = perfect circle).
        bbox: Bounding rectangle (x, y, w, h).
        cell_type: Classification label — ``'RBC'``, ``'WBC'``, or ``'Unknown'``.
        mean_color: Mean BGR colour inside the contour.
        cropped_image: Optional cropped ROI of the cell (BGR).
    """

    cell_id: int
    center: Tuple[int, int]
    radius: int
    area: float
    perimeter: float
    circularity: float
    bbox: Tuple[int, int, int, int]
    cell_type: str
    mean_color: Tuple[float, float, float]
    cropped_image: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ตัวตรวจจับเซลล์หลัก
# ---------------------------------------------------------------------------

class CellDetector:
    """Detect and classify blood cells (RBC / WBC) in microscope images.

    Parameters:
        min_area: Minimum contour area to accept as a cell (pixels²).
        max_area: Maximum contour area to accept as a cell (pixels²).
        circularity_threshold: Minimum circularity (0–1) to accept.
        clahe_clip_limit: CLAHE clip‑limit for contrast enhancement.
        clahe_tile_size: CLAHE tile grid size.
        blur_kernel_size: Gaussian blur kernel size (must be odd).
        morph_kernel_size: Morphological operation kernel size.
        wbc_min_area: Cells with area ≥ this value are WBC candidates.
    """

    # สี สำหรับวาดผลลัพธ์
    _COLOR_RBC: Tuple[int, int, int] = (0, 255, 0)    # Green
    _COLOR_WBC: Tuple[int, int, int] = (255, 0, 0)    # Blue (BGR)
    _COLOR_UNKNOWN: Tuple[int, int, int] = (0, 255, 255)  # Yellow

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 15000,
        circularity_threshold: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        blur_kernel_size: int = 5,
        morph_kernel_size: int = 5,
        wbc_min_area: int = 3000,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.wbc_min_area = wbc_min_area

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_cells(self, image: np.ndarray) -> List[CellInfo]:
        """Run the full detection pipeline on a BGR image.

        Steps:
            1. Convert to grayscale & enhance contrast (CLAHE).
            2. Create binary mask via adaptive thresholding + morphology.
            3. Apply watershed to separate touching / overlapping cells.
            4. Extract contours, compute metrics, and classify each cell.

        Args:
            image: Input BGR image (``np.ndarray`` with shape H×W×3).

        Returns:
            List of :class:`CellInfo` objects for every detected cell.
        """
        if image is None or image.size == 0:
            return []

        # ให้แน่ใจว่าเป็นภาพ BGR 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        preprocessed = self._preprocess(image)
        mask = self._create_mask(preprocessed)
        labels = self._apply_watershed(image, mask)
        cells = self._extract_cells(image, labels)
        return cells

    def draw_detections(
        self,
        image: np.ndarray,
        cells: List[CellInfo],
        *,
        draw_bbox: bool = False,
        font_scale: float = 0.5,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection results on a copy of the image.

        - **RBC** → green circle
        - **WBC** → blue circle
        - **Unknown** → yellow circle

        Each cell is labelled with its ``cell_id`` and type.

        Args:
            image: Original BGR image.
            cells: Detection results from :meth:`detect_cells`.
            draw_bbox: If *True*, also draw the bounding rectangle.
            font_scale: Font scale for labels.
            thickness: Line thickness.

        Returns:
            Annotated BGR image (copy — original is not modified).
        """
        output = image.copy()

        for cell in cells:
            if cell.cell_type == "RBC":
                colour = self._COLOR_RBC
            elif cell.cell_type == "WBC":
                colour = self._COLOR_WBC
            else:
                colour = self._COLOR_UNKNOWN

            # วาดวงกลม
            cv2.circle(output, cell.center, cell.radius, colour, thickness)

            # วาดกรอบ bounding box (ถ้าต้องการ)
            if draw_bbox:
                x, y, w, h = cell.bbox
                cv2.rectangle(output, (x, y), (x + w, y + h), colour, 1)

            # ป้ายกำกับ
            label = f"#{cell.cell_id} {cell.cell_type}"
            label_pos = (cell.center[0] - cell.radius, cell.center[1] - cell.radius - 5)
            cv2.putText(
                output,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                colour,
                1,
                cv2.LINE_AA,
            )

        return output

    def get_cell_statistics(self, cells: List[CellInfo]) -> Dict[str, object]:
        """Compute summary statistics for a list of detected cells.

        Returns:
            Dictionary with keys:
                - ``total_cells``
                - ``rbc_count``, ``wbc_count``, ``unknown_count``
                - ``rbc_percentage``, ``wbc_percentage``
                - ``avg_area``, ``avg_radius``, ``avg_circularity``
                - ``min_area``, ``max_area``
                - ``avg_rbc_area``, ``avg_wbc_area``
        """
        total = len(cells)
        if total == 0:
            return {
                "total_cells": 0,
                "rbc_count": 0,
                "wbc_count": 0,
                "unknown_count": 0,
                "rbc_percentage": 0.0,
                "wbc_percentage": 0.0,
                "avg_area": 0.0,
                "avg_radius": 0.0,
                "avg_circularity": 0.0,
                "min_area": 0.0,
                "max_area": 0.0,
                "avg_rbc_area": 0.0,
                "avg_wbc_area": 0.0,
            }

        rbc_cells = [c for c in cells if c.cell_type == "RBC"]
        wbc_cells = [c for c in cells if c.cell_type == "WBC"]
        unknown_cells = [c for c in cells if c.cell_type == "Unknown"]

        areas = [c.area for c in cells]

        return {
            "total_cells": total,
            "rbc_count": len(rbc_cells),
            "wbc_count": len(wbc_cells),
            "unknown_count": len(unknown_cells),
            "rbc_percentage": round(len(rbc_cells) / total * 100, 2),
            "wbc_percentage": round(len(wbc_cells) / total * 100, 2),
            "avg_area": round(float(np.mean(areas)), 2),
            "avg_radius": round(float(np.mean([c.radius for c in cells])), 2),
            "avg_circularity": round(float(np.mean([c.circularity for c in cells])), 4),
            "min_area": round(float(np.min(areas)), 2),
            "max_area": round(float(np.max(areas)), 2),
            "avg_rbc_area": round(float(np.mean([c.area for c in rbc_cells])), 2) if rbc_cells else 0.0,
            "avg_wbc_area": round(float(np.mean([c.area for c in wbc_cells])), 2) if wbc_cells else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale, apply Gaussian blur and CLAHE.

        CLAHE (Contrast Limited Adaptive Histogram Equalisation) handles
        uneven illumination that is common in microscope images.

        Args:
            image: BGR input image.

        Returns:
            Enhanced grayscale image (``uint8``).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur เพื่อลด noise
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )

        # CLAHE เพื่อปรับคอนทราสต์เฉพาะจุด
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )
        enhanced = clahe.apply(blurred)

        return enhanced

    def _create_mask(self, preprocessed: np.ndarray) -> np.ndarray:
        """Create a clean binary mask from the preprocessed grayscale image.

        Uses adaptive thresholding (handles uneven illumination) followed by
        morphological opening (remove small noise) and closing (fill gaps).

        Args:
            preprocessed: Grayscale image from :meth:`_preprocess`.

        Returns:
            Binary mask (``uint8``, values 0 or 255).
        """
        # Adaptive threshold — ใช้ Gaussian weighted mean เพื่อรับมือกับ
        # แสงที่ไม่สม่ำเสมอ
        binary = cv2.adaptiveThreshold(
            preprocessed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )

        # Opening — ลบจุด noise เล็ก ๆ
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Closing — อุดรูเล็ก ๆ ภายในเซลล์
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

        return closed

    def _apply_watershed(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Separate touching / overlapping cells using the watershed algorithm.

        Steps:
            1. Distance transform on the binary mask.
            2. Threshold the distance map to find *sure foreground* regions
               (peaks = cell centres).
            3. Dilate the mask to find *sure background*.
            4. The difference is the *unknown* region.
            5. Connected‑component labelling on sure foreground → markers.
            6. Run ``cv2.watershed`` to assign every pixel a label.

        Args:
            image: Original BGR image (needed by ``cv2.watershed``).
            mask: Binary mask from :meth:`_create_mask`.

        Returns:
            Label matrix (``int32``).  Background = 1, boundaries = -1,
            each cell gets a unique label ≥ 2.
        """
        # Distance transform — ระยะห่างจากพิกเซลถึงขอบใกล้สุด
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Normalise เพื่อหา peak
        dist_transform = cv2.normalize(
            dist_transform, None, 0, 1.0, cv2.NORM_MINMAX  # type: ignore[arg-type]
        )

        # Sure foreground — เฉพาะจุดที่ห่างจากขอบมาก (ศูนย์กลางเซลล์)
        _, sure_fg = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg * 255)

        # Sure background — ขยาย mask ออก
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        # Unknown region = sure_bg − sure_fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Connected components สำหรับ markers
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # เลื่อน label ขึ้น 1 เพื่อให้ background = 1 (watershed ต้องการ 0 = unknown)
        markers = markers + 1

        # Unknown region ตั้งเป็น 0
        markers[unknown == 255] = 0

        # Watershed
        markers = cv2.watershed(image, markers)

        return markers

    def _extract_cells(
        self, image: np.ndarray, labels: np.ndarray
    ) -> List[CellInfo]:
        """Extract :class:`CellInfo` for each unique watershed label.

        For every label ≥ 2 (skipping background=1 and boundaries=-1):
            - Build a mask for the label.
            - Find its contour and compute geometric properties.
            - Filter by area and circularity thresholds.
            - Classify as RBC or WBC.

        Args:
            image: Original BGR image.
            labels: Label matrix from :meth:`_apply_watershed`.

        Returns:
            Filtered and classified list of :class:`CellInfo`.
        """
        cells: List[CellInfo] = []
        cell_id_counter = 1
        unique_labels = np.unique(labels)

        for label_val in unique_labels:
            # ข้าม background (1) และ boundary (-1)
            if label_val <= 1:
                continue

            # สร้าง mask สำหรับ label นี้
            cell_mask = np.zeros(labels.shape, dtype=np.uint8)
            cell_mask[labels == label_val] = 255

            # หา contour
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # ใช้ contour ที่ใหญ่ที่สุด
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # กรองตาม area
            if area < self.min_area or area > self.max_area:
                continue

            # คำนวณ circularity
            if perimeter == 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            # กรองตาม circularity
            if circularity < self.circularity_threshold:
                continue

            # Bounding rect & enclosing circle
            x, y, w, h = cv2.boundingRect(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            radius = int(radius)

            # Mean colour ภายใน contour mask
            mean_bgr = cv2.mean(image, mask=cell_mask)[:3]

            # Crop ROI (พร้อมขอบเพิ่ม 5 px)
            pad = 5
            y1 = max(0, y - pad)
            y2 = min(image.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(image.shape[1], x + w + pad)
            cropped = image[y1:y2, x1:x2].copy()

            # จำแนกเป็น RBC หรือ WBC
            cell_type = self._classify_cell_type(cropped, area, mean_bgr)

            cells.append(
                CellInfo(
                    cell_id=cell_id_counter,
                    center=center,
                    radius=radius,
                    area=round(area, 2),
                    perimeter=round(perimeter, 2),
                    circularity=round(circularity, 4),
                    bbox=(x, y, w, h),
                    cell_type=cell_type,
                    mean_color=(round(mean_bgr[0], 2), round(mean_bgr[1], 2), round(mean_bgr[2], 2)),
                    cropped_image=cropped,
                )
            )
            cell_id_counter += 1

        return cells

    def _classify_cell_type(
        self,
        cell_image: np.ndarray,
        area: float,
        mean_bgr: Tuple[float, ...],
    ) -> str:
        """Classify a cell as **RBC**, **WBC**, or **Unknown**.

        Classification heuristics (typical Wright‑stained blood smears):

        * **WBC** (White Blood Cells / เม็ดเลือดขาว):
          - Significantly larger than RBCs (area ≥ ``wbc_min_area``).
          - Contain a dark‑staining nucleus that appears purple/blue in
            Wright stain.  In HSV space the nucleus pixels have *hue* in
            the blue–purple range (~100–160) and relatively *low value*.

        * **RBC** (Red Blood Cells / เม็ดเลือดแดง):
          - Smaller, uniform size.
          - Pinkish‑red colour; HSV hue roughly 0–20 or 160–180.

        Args:
            cell_image: Cropped BGR image of the cell.
            area: Contour area of the cell.
            mean_bgr: Mean BGR colour inside the cell contour.

        Returns:
            ``'RBC'``, ``'WBC'``, or ``'Unknown'``.
        """
        if cell_image is None or cell_image.size == 0:
            return "Unknown"

        # --- Size check ---
        # WBC โดยทั่วไปมีขนาดใหญ่กว่า RBC 2–3 เท่า
        is_large = area >= self.wbc_min_area

        # --- Colour analysis in HSV ---
        hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)

        # หา pixel ที่อยู่ในช่วงสีม่วง / น้ำเงินเข้ม (nucleus ของ WBC)
        # Purple-blue nucleus range ใน HSV
        lower_purple = np.array([100, 40, 30], dtype=np.uint8)
        upper_purple = np.array([160, 255, 220], dtype=np.uint8)
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

        total_pixels = cell_image.shape[0] * cell_image.shape[1]
        purple_ratio = float(cv2.countNonZero(purple_mask)) / max(total_pixels, 1)

        # WBC ควรมีบริเวณ nucleus สีม่วง/น้ำเงินอย่างน้อย ~15 %
        has_purple_nucleus = purple_ratio > 0.15

        # --- Decision ---
        if is_large and has_purple_nucleus:
            return "WBC"
        elif is_large and purple_ratio > 0.08:
            # ขนาดใหญ่ + มีสีม่วงบ้าง → ยังน่าจะเป็น WBC
            return "WBC"
        elif not is_large and not has_purple_nucleus:
            return "RBC"
        else:
            # กรณีคลุมเครือ — ใช้ mean colour ช่วยตัดสิน
            # RBC มีค่า Blue channel ต่ำ (สีแดง/ชมพู)
            mean_b, mean_g, mean_r = mean_bgr[:3]
            if mean_b > mean_r and is_large:
                return "WBC"
            elif mean_r >= mean_b:
                return "RBC"
            else:
                return "Unknown"
