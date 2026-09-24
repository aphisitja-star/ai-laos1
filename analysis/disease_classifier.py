"""disease_classifier — โมดูลตรวจจับโรคมาลาเรียในเซลล์เม็ดเลือดแดง.

ใช้การวิเคราะห์สี (OpenCV) หรือโมเดล CNN เพื่อตรวจหาปรสิตมาลาเรีย
ภายในเซลล์เม็ดเลือดแดงที่ย้อมสี Giemsa.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InfectionResult:
    """Result for a single cell's infection analysis.

    Attributes:
        cell_id: Unique identifier for the cell.
        is_infected: Whether the cell is classified as infected.
        confidence: Detection confidence in the range [0.0, 1.0].
        method: Analysis method used – ``'opencv'`` or ``'cnn'``.
        parasite_regions: List of bounding boxes ``(x, y, w, h)`` for each
            detected parasite spot inside the cell.
    """

    cell_id: int
    is_infected: bool
    confidence: float
    method: str
    parasite_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report for a blood smear image.

    Attributes:
        total_cells: Total number of cells analysed.
        infected_cells: Number of cells classified as infected.
        uninfected_cells: Number of cells classified as uninfected.
        parasitemia_rate: Infection rate expressed as a percentage.
        severity_level: Human-readable severity string (Thai + English).
        severity_color: Hex colour string for UI rendering.
        cell_results: Per-cell infection results.
        analysis_method: ``'opencv'`` or ``'cnn'``.
    """

    total_cells: int
    infected_cells: int
    uninfected_cells: int
    parasitemia_rate: float
    severity_level: str
    severity_color: str
    cell_results: List[InfectionResult]
    analysis_method: str

    @property
    def summary_thai(self) -> str:
        """สร้างข้อความสรุปผลภาษาไทย."""
        return (
            f"พบเซลล์ทั้งหมด {self.total_cells} เซลล์ | "
            f"ติดเชื้อ {self.infected_cells} เซลล์ | "
            f"อัตราการติดเชื้อ {self.parasitemia_rate:.2f}%"
        )


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class DiseaseClassifier:
    """Malaria parasite detector for Giemsa-stained thin blood-smear images.

    The classifier supports two analysis back-ends:

    1. **OpenCV** (default) – colour thresholding in HSV space followed by
       morphological filtering.  No external model required.
    2. **CNN** – a pre-trained TensorFlow/Keras model loaded from disk.

    Parameters:
        model_path: Optional path to a ``*.h5`` / SavedModel directory.
            When provided *and* the file exists the CNN back-end is used
            automatically.
    """

    # --- HSV thresholds for dark-purple / blue Giemsa-stained parasites ---
    # Lower & upper HSV bounds (multiple ranges to cover purple→blue)
    _PARASITE_HSV_RANGES: List[Tuple[np.ndarray, np.ndarray]] = [
        # Dark purple / violet
        (np.array([120, 40, 20]), np.array([160, 255, 150])),
        # Dark blue
        (np.array([100, 40, 20]), np.array([130, 255, 140])),
        # Very dark (almost black) stained spots – low value regardless of hue
        (np.array([100, 20, 0]),  np.array([170, 255, 80])),
    ]

    # Relative-area thresholds (fraction of cell area)
    _MIN_PARASITE_AREA_FRAC: float = 0.005   # parasite ≥ 0.5 % of cell
    _MAX_PARASITE_AREA_FRAC: float = 0.35     # parasite ≤ 35 % of cell
    _INFECTION_DARK_RATIO_THRESH: float = 0.01  # ≥ 1 % dark pixels → infected

    # CNN input dimensions
    _CNN_INPUT_SIZE: Tuple[int, int] = (128, 128)

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = None
        self.model_path = model_path
        self.use_cnn: bool = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load a trained TensorFlow / Keras model from *model_path*."""
        try:
            from tensorflow import keras  # type: ignore[import-untyped]

            self.model = keras.models.load_model(model_path)
            self.use_cnn = True
        except Exception as exc:  # pragma: no cover
            print(f"[DiseaseClassifier] Could not load model: {exc}")
            self.model = None
            self.use_cnn = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image: np.ndarray,
        cells: list,
    ) -> AnalysisReport:
        """Run the full analysis pipeline and return an :class:`AnalysisReport`.

        Parameters:
            image: BGR source image (as returned by ``cv2.imread``).
            cells: Sequence of detected cells.  Each element must expose
                ``x, y, w, h`` attributes **or** be a 4-tuple/list of
                ``(x, y, w, h)``.

        Returns:
            An :class:`AnalysisReport` summarising infection status.
        """
        if self.use_cnn and self.model is not None:
            results = self._analyze_with_cnn(image, cells)
            method = "cnn"
        else:
            results = self._analyze_with_opencv(image, cells)
            method = "opencv"

        infected = sum(1 for r in results if r.is_infected)
        total = len(results)
        uninfected = total - infected
        rate = (infected / total * 100.0) if total > 0 else 0.0
        severity_level, severity_color = self._get_severity(rate)

        return AnalysisReport(
            total_cells=total,
            infected_cells=infected,
            uninfected_cells=uninfected,
            parasitemia_rate=rate,
            severity_level=severity_level,
            severity_color=severity_color,
            cell_results=results,
            analysis_method=method,
        )

    # ------------------------------------------------------------------
    # OpenCV-based analysis
    # ------------------------------------------------------------------

    def _analyze_with_opencv(
        self,
        image: np.ndarray,
        cells: list,
    ) -> List[InfectionResult]:
        """Analyse cells with OpenCV colour / morphology heuristics.

        For every detected cell region the method:
        1. Crops the cell from the source image.
        2. Calls :meth:`_detect_parasites_in_cell` to look for dark
           purple / blue Giemsa-stained spots.
        3. Assembles an :class:`InfectionResult`.
        """
        results: List[InfectionResult] = []

        for idx, cell in enumerate(cells):
            x, y, w, h = self._unpack_cell(cell)

            # Clamp to image boundaries
            img_h, img_w = image.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            if x2 - x1 < 5 or y2 - y1 < 5:
                # Cell region too small – skip
                results.append(
                    InfectionResult(
                        cell_id=idx,
                        is_infected=False,
                        confidence=0.0,
                        method="opencv",
                        parasite_regions=[],
                    )
                )
                continue

            cell_crop = image[y1:y2, x1:x2].copy()
            is_infected, confidence, parasite_bboxes = (
                self._detect_parasites_in_cell(cell_crop)
            )

            # Offset bounding boxes back to full-image coordinates
            global_bboxes = [
                (bx + x1, by + y1, bw, bh)
                for bx, by, bw, bh in parasite_bboxes
            ]

            results.append(
                InfectionResult(
                    cell_id=idx,
                    is_infected=is_infected,
                    confidence=confidence,
                    method="opencv",
                    parasite_regions=global_bboxes,
                )
            )

        return results

    # ------------------------------------------------------------------
    # CNN-based analysis
    # ------------------------------------------------------------------

    def _analyze_with_cnn(
        self,
        image: np.ndarray,
        cells: list,
    ) -> List[InfectionResult]:
        """Analyse cells using a trained CNN (TensorFlow / Keras).

        Steps
        -----
        1. Crop each cell from the source image.
        2. Resize to the expected model input dimensions.
        3. Normalise pixel values to ``[0, 1]``.
        4. Batch-predict with the model.
        5. Interpret model output as ``P(infected)``.
        """
        if self.model is None:
            # Fallback – should not happen because caller checks use_cnn
            return self._analyze_with_opencv(image, cells)

        img_h, img_w = image.shape[:2]
        input_h, input_w = self._CNN_INPUT_SIZE
        batch: List[np.ndarray] = []
        valid_indices: List[int] = []

        for idx, cell in enumerate(cells):
            x, y, w, h = self._unpack_cell(cell)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = image[y1:y2, x1:x2]
            resized = cv2.resize(crop, (input_w, input_h))
            normalised = resized.astype(np.float32) / 255.0
            batch.append(normalised)
            valid_indices.append(idx)

        results: List[InfectionResult] = []

        if batch:
            batch_arr = np.array(batch)
            predictions = self.model.predict(batch_arr, verbose=0)

            # Handle both single-output (sigmoid) and two-class (softmax)
            if predictions.shape[-1] == 1:
                probs = predictions.ravel()
            else:
                # Column 1 = P(infected)
                probs = predictions[:, 1]

            pred_map = dict(zip(valid_indices, probs))
        else:
            pred_map = {}

        for idx in range(len(cells)):
            if idx in pred_map:
                prob = float(pred_map[idx])
                infected = prob >= 0.5
                results.append(
                    InfectionResult(
                        cell_id=idx,
                        is_infected=infected,
                        confidence=prob if infected else 1.0 - prob,
                        method="cnn",
                        parasite_regions=[],
                    )
                )
            else:
                results.append(
                    InfectionResult(
                        cell_id=idx,
                        is_infected=False,
                        confidence=0.0,
                        method="cnn",
                        parasite_regions=[],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Core parasite detection (OpenCV)
    # ------------------------------------------------------------------

    def _detect_parasites_in_cell(
        self,
        cell_image: np.ndarray,
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]]]:
        """Detect parasite regions within a single cropped cell image.

        Algorithm
        ---------
        1. Convert to HSV colour space.
        2. Build a combined mask using multiple HSV ranges that capture the
           dark purple / blue tones typical of Giemsa-stained *Plasmodium*
           parasites.
        3. Apply morphological *opening* (remove noise) then *closing*
           (fill small gaps).
        4. Find contours in the cleaned mask and filter by area relative
           to the cell size.
        5. Calculate a confidence score from the ratio of dark-stained
           pixels to total cell area and the number / quality of detected
           spots.

        Parameters:
            cell_image: BGR crop of a single cell.

        Returns:
            ``(is_infected, confidence, parasite_bboxes)``
        """
        h, w = cell_image.shape[:2]
        cell_area = h * w

        if cell_area == 0:
            return False, 0.0, []

        # --- 1. Colour-space conversion ------------------------------------
        hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)

        # --- 2. Build combined parasite mask --------------------------------
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self._PARASITE_HSV_RANGES:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Additionally detect very dark pixels regardless of hue (catches
        # heavily-stained trophozoites / schizonts that appear near-black).
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Intersect dark mask with a broad saturation filter to avoid
        # confusing cell-centre pallor (which is light, not saturated)
        # with parasites.
        sat_channel = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(sat_channel, 30, 255, cv2.THRESH_BINARY)
        dark_and_coloured = cv2.bitwise_and(dark_mask, sat_mask)
        combined_mask = cv2.bitwise_or(combined_mask, dark_and_coloured)

        # --- 3. Morphological clean-up --------------------------------------
        # Use an elliptical kernel sized proportionally to the cell
        kernel_size = max(3, int(min(h, w) * 0.04) | 1)  # ensure odd
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # --- 4. Exclude cell border artefacts --------------------------------
        # Create a circular mask to keep only the inner ~80 % of the cell
        # (edges often have staining artefacts).
        centre = (w // 2, h // 2)
        radius = int(min(h, w) * 0.40)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_mask, centre, radius, 255, thickness=-1)
        combined_mask = cv2.bitwise_and(combined_mask, inner_mask)

        # --- 5. Find contours & filter by area ------------------------------
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = int(cell_area * self._MIN_PARASITE_AREA_FRAC)
        max_area = int(cell_area * self._MAX_PARASITE_AREA_FRAC)

        parasite_bboxes: List[Tuple[int, int, int, int]] = []
        total_parasite_pixels = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                parasite_bboxes.append((bx, by, bw, bh))
                total_parasite_pixels += area

        # --- 6. Decision & confidence --------------------------------------
        dark_ratio = total_parasite_pixels / cell_area
        is_infected = (
            dark_ratio >= self._INFECTION_DARK_RATIO_THRESH
            and len(parasite_bboxes) > 0
        )

        if is_infected:
            # Confidence increases with dark-ratio (capped at 1.0)
            # and with number of detected spots (diminishing returns).
            ratio_score = min(dark_ratio / 0.10, 1.0)  # saturates at 10 %
            count_score = min(len(parasite_bboxes) / 3.0, 1.0)
            confidence = 0.5 + 0.3 * ratio_score + 0.2 * count_score
            confidence = float(np.clip(confidence, 0.0, 1.0))
        else:
            confidence = float(np.clip(1.0 - dark_ratio * 10, 0.5, 1.0))

        return is_infected, confidence, parasite_bboxes

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _get_severity(parasitemia_rate: float) -> Tuple[str, str]:
        """Return ``(severity_label, hex_colour)`` based on *parasitemia_rate*.

        WHO-aligned rough thresholds:
        * < 1 %  → ต่ำ (Low)         — green
        * 1–5 %  → ปานกลาง (Moderate) — amber / yellow
        * ≥ 5 %  → สูง (High)         — red
        """
        if parasitemia_rate < 1.0:
            return ("ต่ำ (Low)", "#28a745")
        elif parasitemia_rate < 5.0:
            return ("ปานกลาง (Moderate)", "#ffc107")
        else:
            return ("สูง (High)", "#dc3545")

    # ------------------------------------------------------------------
    # Heatmap visualisation
    # ------------------------------------------------------------------

    def create_heatmap(
        self,
        image: np.ndarray,
        cell_results: List[InfectionResult],
        cells: list,
    ) -> np.ndarray:
        """Create a translucent heatmap overlay on *image*.

        Infected cells are highlighted in **red**; healthy cells in
        **green**.  The overlay intensity scales with the detection
        confidence.

        Parameters:
            image: Original BGR image.
            cell_results: List of :class:`InfectionResult` objects.
            cells: Corresponding cell descriptors (same order as
                *cell_results*).

        Returns:
            BGR image with the heatmap overlay blended on top.
        """
        overlay = image.copy()
        output = image.copy()

        for result, cell in zip(cell_results, cells):
            x, y, w, h = self._unpack_cell(cell)

            # Colour: red for infected, green for healthy
            if result.is_infected:
                colour = (0, 0, 255)  # BGR red
            else:
                colour = (0, 200, 0)  # BGR green

            alpha = 0.15 + 0.25 * result.confidence  # 0.15 – 0.40

            # Draw a filled ellipse matching the cell area
            centre = (x + w // 2, y + h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(overlay, centre, axes, 0, 0, 360, colour, thickness=-1)

            # Draw parasite bounding boxes for infected cells
            if result.is_infected:
                for bx, by, bw, bh in result.parasite_regions:
                    cv2.rectangle(
                        overlay,
                        (bx, by),
                        (bx + bw, by + bh),
                        (0, 0, 255),
                        thickness=2,
                    )

            # Blend this cell's overlay region
            x1, y1 = max(0, x - 2), max(0, y - 2)
            x2 = min(image.shape[1], x + w + 2)
            y2 = min(image.shape[0], y + h + 2)
            output[y1:y2, x1:x2] = cv2.addWeighted(
                overlay[y1:y2, x1:x2],
                alpha,
                output[y1:y2, x1:x2],
                1.0 - alpha,
                0,
            )

        # Add a small legend in the top-left corner
        legend_y = 20
        cv2.putText(
            output,
            "Infected",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(output, (90, legend_y - 4), 5, (0, 0, 255), -1)
        legend_y += 20
        cv2.putText(
            output,
            "Healthy",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(output, (82, legend_y - 4), 5, (0, 200, 0), -1)

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_cell(cell) -> Tuple[int, int, int, int]:
        """Extract ``(x, y, w, h)`` from various cell representations.

        Supports:
        * ``CellInfo`` dataclass objects (with ``bbox`` attribute).
        * Objects with ``x, y, w, h`` attributes.
        * 4-element tuples / lists.
        * Dicts with keys ``'x', 'y', 'w', 'h'`` or ``'bbox'``.
        """
        # CellInfo dataclass — bbox is (x, y, w, h)
        if hasattr(cell, "bbox"):
            bx = cell.bbox
            return int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        elif hasattr(cell, "x"):
            return int(cell.x), int(cell.y), int(cell.w), int(cell.h)
        elif isinstance(cell, dict):
            if "bbox" in cell:
                bx = cell["bbox"]
                return int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            return (
                int(cell["x"]),
                int(cell["y"]),
                int(cell["w"]),
                int(cell["h"]),
            )
        else:
            # Assume sequence (tuple / list)
            return int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
