"""
โมดูลประมวลผลภาพ (Image Processor Module)
==========================================

จัดการการประมวลผลภาพเบื้องต้นทั้งหมดสำหรับระบบวิเคราะห์เซลล์เม็ดเลือด
รองรับการปรับปรุงคุณภาพภาพ, การลดสัญญาณรบกวน, การแปลงปริภูมิสี,
การแยกสี (stain separation) และการปรับขนาดภาพสำหรับโมเดล CNN

Handles all image preprocessing for the blood cell analysis pipeline.
Supports contrast enhancement, noise reduction, color space conversions,
stain separation (approximate H&E deconvolution), and model-ready resizing.
"""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageProcessor:
    """ตัวประมวลผลภาพสำหรับระบบวิเคราะห์เซลล์เม็ดเลือด

    Provides a fluent (method-chaining) API for building an image-processing
    pipeline.  Each mutating method modifies ``self.processed`` in-place and
    returns ``self`` so that calls can be chained::

        result = (
            ImageProcessor(image)
            .enhance_contrast()
            .reduce_noise(method='median')
            .normalize()
            .get_result()
        )

    Attributes:
        original: สำเนาภาพต้นฉบับที่ไม่ถูกแก้ไข (immutable copy of the input).
        processed: ภาพที่กำลังถูกประมวลผล (working copy modified by each step).
    """

    # ------------------------------------------------------------------
    # H&E stain vectors (optical-density space, normalised)
    # Reference: Ruifrok & Johnston, "Quantification of histochemical
    # staining by color deconvolution", Anal. Quant. Cytol. Histol., 2001.
    # ------------------------------------------------------------------
    _HE_STAIN_MATRIX: NDArray[np.float64] = np.array(
        [
            [0.644211, 0.716556, 0.266844],  # Hematoxylin (purple/blue)
            [0.092789, 0.954111, 0.283111],  # Eosin       (pink/red)
            [0.0,      0.0,      0.0     ],  # Residual    (unused)
        ],
        dtype=np.float64,
    )

    def __init__(self, image: NDArray[np.uint8]) -> None:
        """สร้าง ImageProcessor จากภาพ BGR

        Args:
            image: ภาพ BGR ขนาด (H, W, 3) ชนิด ``np.uint8``

        Raises:
            ValueError: ถ้าภาพไม่ใช่ 3-channel หรือ dtype ไม่ใช่ uint8
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"ต้องการภาพ 3-channel (BGR) แต่ได้รับ shape={image.shape}"
            )
        self.original: NDArray[np.uint8] = image.copy()
        self.processed: NDArray[np.uint8] = image.copy()

    # ==================================================================
    # Enhancement / filtering
    # ==================================================================

    def enhance_contrast(
        self,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8),
    ) -> ImageProcessor:
        """ปรับปรุงคอนทราสต์ด้วย CLAHE (Contrast Limited Adaptive Histogram Equalisation)

        ทำงานบนช่อง L ของปริภูมิสี CIELAB เพื่อปรับความสว่างโดยไม่
        เปลี่ยนแปลงเฉดสี

        Args:
            clip_limit: ขีดจำกัดคอนทราสต์สำหรับ CLAHE (ค่ามาก = คอนทราสต์สูง)
            tile_size: ขนาดตาราง (grid) สำหรับการคำนวณ histogram ท้องถิ่น

        Returns:
            self — เพื่อรองรับ method chaining
        """
        lab: NDArray[np.uint8] = cv2.cvtColor(self.processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size,
        )
        l_enhanced: NDArray[np.uint8] = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        self.processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return self

    def reduce_noise(
        self,
        method: Literal["gaussian", "median"] = "gaussian",
        kernel_size: int = 5,
    ) -> ImageProcessor:
        """ลดสัญญาณรบกวน (noise reduction)

        Args:
            method: ``'gaussian'`` สำหรับ Gaussian blur หรือ
                     ``'median'`` สำหรับ Median blur
            kernel_size: ขนาดเคอร์เนล (ต้องเป็นเลขคี่)

        Returns:
            self — เพื่อรองรับ method chaining

        Raises:
            ValueError: ถ้า *method* ไม่ใช่ ``'gaussian'`` หรือ ``'median'``
                         หรือ *kernel_size* เป็นเลขคู่
        """
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size ต้องเป็นเลขคี่ แต่ได้รับ {kernel_size}"
            )

        if method == "gaussian":
            self.processed = cv2.GaussianBlur(
                self.processed, (kernel_size, kernel_size), 0
            )
        elif method == "median":
            self.processed = cv2.medianBlur(self.processed, kernel_size)
        else:
            raise ValueError(
                f"method ต้องเป็น 'gaussian' หรือ 'median' แต่ได้รับ '{method}'"
            )
        return self

    def normalize(self) -> ImageProcessor:
        """ทำ Normalisation ค่าพิกเซลให้อยู่ในช่วง [0, 1]

        แปลง dtype จาก ``uint8`` เป็น ``float32`` และหารด้วย 255

        Returns:
            self — เพื่อรองรับ method chaining
        """
        self.processed = self.processed.astype(np.float32) / 255.0  # type: ignore[assignment]
        return self

    # ==================================================================
    # Color-space conversions (return a *new* array, do NOT modify self)
    # ==================================================================

    def to_grayscale(self) -> NDArray[np.uint8]:
        """แปลงภาพปัจจุบันเป็นภาพ Grayscale

        Returns:
            ภาพ Grayscale ขนาด (H, W)
        """
        return cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)

    def to_hsv(self) -> NDArray[np.uint8]:
        """แปลงภาพปัจจุบันเป็นปริภูมิสี HSV

        Returns:
            ภาพ HSV ขนาด (H, W, 3)
        """
        return cv2.cvtColor(self.processed, cv2.COLOR_BGR2HSV)

    def to_lab(self) -> NDArray[np.uint8]:
        """แปลงภาพปัจจุบันเป็นปริภูมิสี CIELAB

        Returns:
            ภาพ LAB ขนาด (H, W, 3)
        """
        return cv2.cvtColor(self.processed, cv2.COLOR_BGR2LAB)

    # ==================================================================
    # Thresholding
    # ==================================================================

    def apply_threshold(
        self,
        method: Literal["adaptive", "otsu", "binary"] = "adaptive",
        binary_thresh: int = 127,
        max_value: int = 255,
        block_size: int = 11,
        c_value: int = 2,
    ) -> NDArray[np.uint8]:
        """ทำ Thresholding บนภาพปัจจุบัน

        Args:
            method:
                * ``'adaptive'`` — Adaptive Gaussian threshold
                * ``'otsu'``     — Otsu's automatic threshold
                * ``'binary'``   — Fixed-value binary threshold
            binary_thresh: ค่า threshold สำหรับ method ``'binary'`` (0–255)
            max_value: ค่าสูงสุดที่กำหนดให้พิกเซลที่ผ่านเกณฑ์
            block_size: ขนาดบล็อกสำหรับ adaptive threshold (ต้องเป็นเลขคี่)
            c_value: ค่าคงที่ที่หักออกจาก mean ในกรณี adaptive

        Returns:
            ภาพ binary (H, W) ชนิด ``uint8``

        Raises:
            ValueError: ถ้า *method* ไม่ถูกต้อง
        """
        gray: NDArray[np.uint8] = self.to_grayscale()

        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray,
                max_value,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value,
            )
        elif method == "otsu":
            _, thresh = cv2.threshold(
                gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh
        elif method == "binary":
            _, thresh = cv2.threshold(
                gray, binary_thresh, max_value, cv2.THRESH_BINARY
            )
            return thresh
        else:
            raise ValueError(
                f"method ต้องเป็น 'adaptive', 'otsu', หรือ 'binary' "
                f"แต่ได้รับ '{method}'"
            )

    # ==================================================================
    # Stain separation (color deconvolution)
    # ==================================================================

    def get_stain_separated(
        self,
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """แยกสีย้อม (Stain Separation) โดยประมาณด้วย Color Deconvolution

        ใช้เวกเตอร์สีย้อม H&E มาตรฐานเพื่อแยกช่อง Hematoxylin (ม่วง/น้ำเงิน)
        และ Eosin (ชมพู/แดง) ออกจากกัน เหมาะสำหรับภาพสเมียร์เลือด

        Returns:
            ``(hematoxylin, eosin)`` — แต่ละช่องเป็นภาพ Grayscale
            ``uint8`` ขนาด (H, W)
        """
        # --- Build an invertible 3×3 stain matrix ---
        stain_matrix = self._HE_STAIN_MATRIX.copy()

        # Replace zero-row with a perpendicular residual vector so the
        # matrix is invertible.
        if np.all(stain_matrix[2] == 0):
            stain_matrix[2] = np.cross(stain_matrix[0], stain_matrix[1])
            norm = np.linalg.norm(stain_matrix[2])
            if norm > 1e-6:
                stain_matrix[2] /= norm
            else:
                # Fallback: use an arbitrary orthogonal direction
                stain_matrix[2] = np.array([1.0, 0.0, 0.0])

        deconv_matrix: NDArray[np.float64] = np.linalg.inv(stain_matrix)

        # --- Convert BGR → RGB and into optical-density space ---
        rgb = cv2.cvtColor(self.processed, cv2.COLOR_BGR2RGB).astype(np.float64)
        rgb = np.clip(rgb, 1.0, 255.0)  # avoid log(0)
        optical_density: NDArray[np.float64] = -np.log(rgb / 255.0)

        # --- Deconvolve ---
        # Reshape to (N, 3), multiply, reshape back
        h, w, _ = optical_density.shape
        od_flat = optical_density.reshape(-1, 3)  # (N, 3)
        stain_channels = od_flat @ deconv_matrix.T  # (N, 3)
        stain_channels = stain_channels.reshape(h, w, 3)

        # --- Extract & rescale individual stain channels to uint8 ---
        def _to_uint8(channel: NDArray[np.float64]) -> NDArray[np.uint8]:
            """Rescale a single OD-channel to 0-255 uint8."""
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min < 1e-6:
                return np.zeros_like(channel, dtype=np.uint8)
            normalised = (channel - c_min) / (c_max - c_min) * 255.0
            return normalised.astype(np.uint8)

        hematoxylin: NDArray[np.uint8] = _to_uint8(stain_channels[:, :, 0])
        eosin: NDArray[np.uint8] = _to_uint8(stain_channels[:, :, 1])

        return hematoxylin, eosin

    # ==================================================================
    # Model helpers
    # ==================================================================

    def resize_for_model(
        self,
        target_size: Tuple[int, int] = (224, 224),
    ) -> NDArray[np.uint8]:
        """ปรับขนาดภาพให้พร้อมสำหรับป้อนเข้าโมเดล CNN

        ใช้ ``cv2.INTER_AREA`` เมื่อย่อขนาด และ ``cv2.INTER_LINEAR``
        เมื่อขยายขนาด เพื่อให้ได้คุณภาพที่ดีที่สุด

        Args:
            target_size: ``(width, height)`` ของภาพเอาต์พุต

        Returns:
            ภาพที่ปรับขนาดแล้ว (H, W, 3) หรือ (H, W) ชนิด ``uint8``
        """
        h, w = self.processed.shape[:2]
        target_w, target_h = target_size

        # เลือก interpolation ให้เหมาะสมกับทิศทางการปรับขนาด
        if target_h * target_w < h * w:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        resized: NDArray[np.uint8] = cv2.resize(
            self.processed,
            (target_w, target_h),
            interpolation=interpolation,
        )
        return resized

    # ==================================================================
    # Result retrieval
    # ==================================================================

    def get_result(self) -> NDArray[np.uint8]:
        """ส่งคืนภาพที่ผ่านการประมวลผลแล้ว

        Returns:
            สำเนาของ ``self.processed``
        """
        return self.processed.copy()

    def reset(self) -> ImageProcessor:
        """รีเซ็ตภาพกลับเป็นต้นฉบับ

        Returns:
            self — เพื่อรองรับ method chaining
        """
        self.processed = self.original.copy()
        return self
