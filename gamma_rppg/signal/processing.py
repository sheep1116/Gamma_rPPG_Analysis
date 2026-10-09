from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from pyVHR.extraction.sig_extraction_methods import holistic_mean
from pyVHR.extraction.sig_processing import SignalProcessing, SignalProcessingParams
from pyVHR.extraction.utils import extract_frames_yield

from gamma_rppg.signal.image_math import apply_gamma_pixels, encoded_luma, make_gamma_table


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _natural_key(path_like):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path_like))]


def iter_frames_inline(video_or_frame_dir):
    """Yield BGR frames from a video file or an image directory."""
    path = Path(str(video_or_frame_dir))
    if path.is_dir():
        frame_files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
            key=_natural_key,
        )
        if not frame_files:
            raise FileNotFoundError(f"No image frames found in directory: {path}")
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Unreadable frame skipped: {frame_path}")
                continue
            yield frame
    else:
        yield from extract_frames_yield(str(video_or_frame_dir))



@dataclass
class BrightnessStats:
    # Availability and ROI size.
    frame_count: int = 0
    face_detected_frames: int = 0
    geometric_roi_frames: int = 0
    valid_signal_frames: int = 0
    roi_area_mean: float = np.nan
    roi_area_std: float = np.nan
    signal_pixel_count_mean: float = np.nan
    signal_pixel_count_std: float = np.nan
    valid_pixel_ratio_mean: float = np.nan
    valid_pixel_ratio_std: float = np.nan
    rejected_dark_pixel_ratio: float = np.nan
    rejected_bright_pixel_ratio: float = np.nan

    # v4-compatible aggregate values.
    mean_rgb: float = np.nan
    std_rgb: float = np.nan
    p95_rgb: float = np.nan
    p99_rgb: float = np.nan
    sat_ratio_250: float = np.nan
    sat_ratio_245: float = np.nan
    dark_ratio_5: float = np.nan
    dark_ratio_10: float = np.nan
    mean_r: float = np.nan
    mean_g: float = np.nan
    mean_b: float = np.nan
    rg_ratio: float = np.nan
    bg_ratio: float = np.nan
    raw_cv_r: float = np.nan
    raw_cv_g: float = np.nan
    raw_cv_b: float = np.nan

    # Explicit RGB/color descriptors.
    roi_mean_r: float = np.nan
    roi_mean_g: float = np.nan
    roi_mean_b: float = np.nan
    roi_median_r: float = np.nan
    roi_median_g: float = np.nan
    roi_median_b: float = np.nan
    r_over_g: float = np.nan
    b_over_g: float = np.nan
    r_over_b: float = np.nan
    r_minus_g: float = np.nan
    g_minus_b: float = np.nan
    channel_range: float = np.nan

    # Encoded-RGB luma descriptors.
    roi_luma_mean: float = np.nan
    roi_luma_median: float = np.nan
    roi_luma_std: float = np.nan
    roi_luma_p05: float = np.nan
    roi_luma_p25: float = np.nan
    roi_luma_p75: float = np.nan
    roi_luma_p95: float = np.nan
    roi_luma_p99: float = np.nan
    roi_luma_iqr: float = np.nan
    roi_luma_p95_minus_p05: float = np.nan
    dark_ratio_20: float = np.nan
    bright_ratio_220: float = np.nan
    bright_ratio_235: float = np.nan

    # Temporal and spatial descriptors.
    temporal_luma_std: float = np.nan
    temporal_luma_cv: float = np.nan
    temporal_r_std: float = np.nan
    temporal_g_std: float = np.nan
    temporal_b_std: float = np.nan
    temporal_r_cv: float = np.nan
    temporal_g_cv: float = np.nan
    temporal_b_cv: float = np.nan
    mean_frame_to_frame_luma_change: float = np.nan
    p95_frame_to_frame_luma_change: float = np.nan
    spatial_luma_std: float = np.nan
    spatial_luma_iqr: float = np.nan
    highlight_component_ratio: float = np.nan
    shadow_component_ratio: float = np.nan
    left_right_luma_difference: float = np.nan
    upper_lower_luma_difference: float = np.nan

    # Signal-valid pixels after pyVHR thresholding.
    valid_mean_rgb: float = np.nan
    valid_std_rgb: float = np.nan
    valid_mean_r: float = np.nan
    valid_mean_g: float = np.nan
    valid_mean_b: float = np.nan
    valid_pixel_luma_mean: float = np.nan
    valid_pixel_luma_std: float = np.nan

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class GammaSignalProcessing(SignalProcessing):
    """pyVHR SignalProcessing with an in-memory pre-ROI Gamma transform."""

    def __init__(
        self,
        gamma_value: Optional[float] = None,
        transform_mode: str = "pre_roi_gamma",
        collect_brightness: bool = True,
        example_output_path: Optional[str] = None,
    ):
        super().__init__()
        aliases = {"all": "pre_roi_gamma"}
        self.transform_mode = aliases.get(transform_mode, transform_mode)
        if self.transform_mode != "pre_roi_gamma":
            raise ValueError("GammaSignalProcessing in v5 is reserved for pre_roi_gamma")
        self.collect_brightness = collect_brightness
        self.gamma_value = gamma_value
        self.gamma_table = None
        self.last_brightness_stats: Dict[str, float] = {}
        self.example_output_path = Path(example_output_path) if example_output_path else None
        self._example_saved = False
        if gamma_value is not None:
            self.set_gamma(gamma_value)

    def set_gamma(self, gamma_value: float):
        self.gamma_value = float(gamma_value)
        self.gamma_table = make_gamma_table(self.gamma_value)

    def _apply_gamma(self, rgb_image: np.ndarray) -> np.ndarray:
        return rgb_image if self.gamma_table is None else cv2.LUT(rgb_image, self.gamma_table)

    @staticmethod
    def _geometry_mask(cropped_skin_im: np.ndarray, cropped_mask_im: Optional[np.ndarray]) -> np.ndarray:
        if cropped_skin_im is None or cropped_skin_im.size == 0:
            return np.zeros((0, 0), dtype=bool)
        if cropped_mask_im is None or cropped_mask_im.shape[:2] != cropped_skin_im.shape[:2]:
            return np.any(cropped_skin_im > 0, axis=2)
        return np.any(cropped_mask_im > 0, axis=2)

    @classmethod
    def _geometry_pixels(cls, cropped_skin_im: np.ndarray, cropped_mask_im: Optional[np.ndarray]) -> np.ndarray:
        mask = cls._geometry_mask(cropped_skin_im, cropped_mask_im)
        if mask.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return cropped_skin_im[mask].reshape(-1, 3).astype(np.float32)

    @staticmethod
    def _signal_valid_pixels(geometry_pixels: np.ndarray) -> np.ndarray:
        if geometry_pixels.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        lo = int(SignalProcessingParams.RGB_LOW_TH)
        hi = int(SignalProcessingParams.RGB_HIGH_TH)
        in_range = np.all((geometry_pixels >= lo) & (geometry_pixels <= hi), axis=1)
        return geometry_pixels[in_range].astype(np.float32)

    @staticmethod
    def _bounded_sample(pixels: np.ndarray, maximum: int = 5000) -> np.ndarray:
        if pixels.shape[0] <= maximum:
            return pixels
        indices = np.linspace(0, pixels.shape[0] - 1, maximum).astype(np.int64)
        return pixels[indices]

    @classmethod
    def _spatial_metrics(
        cls, cropped_skin_im: np.ndarray, cropped_mask_im: Optional[np.ndarray]
    ) -> Dict[str, float]:
        mask = cls._geometry_mask(cropped_skin_im, cropped_mask_im)
        if mask.size == 0 or not np.any(mask):
            return {}
        luma_image = encoded_luma(cropped_skin_im)
        values = luma_image[mask]
        ys, xs = np.where(mask)
        xmid = (int(xs.min()) + int(xs.max())) / 2.0
        ymid = (int(ys.min()) + int(ys.max())) / 2.0
        left = values[xs <= xmid]
        right = values[xs > xmid]
        upper = values[ys <= ymid]
        lower = values[ys > ymid]
        return {
            "spatial_luma_std": float(np.std(values)),
            "spatial_luma_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "highlight_component_ratio": float(np.mean(values >= 235)),
            "shadow_component_ratio": float(np.mean(values <= 20)),
            "left_right_luma_difference": float(abs(np.mean(left) - np.mean(right))) if left.size and right.size else np.nan,
            "upper_lower_luma_difference": float(abs(np.mean(upper) - np.mean(lower))) if upper.size and lower.size else np.nan,
        }

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return float(np.mean(array)) if array.size else np.nan

    @classmethod
    def _summarize_brightness(
        cls,
        geometry_frames: List[np.ndarray],
        valid_frames: List[np.ndarray],
        geometry_areas: List[int],
        valid_counts: List[int],
        face_detected_frames: int,
        frame_rgb_means: List[np.ndarray],
        frame_luma_means: List[float],
        spatial_rows: List[Dict[str, float]],
        rejected_dark_counts: List[int],
        rejected_bright_counts: List[int],
    ) -> BrightnessStats:
        stats = BrightnessStats()
        stats.frame_count = len(geometry_areas)
        stats.face_detected_frames = int(face_detected_frames)
        stats.geometric_roi_frames = int(sum(area > 0 for area in geometry_areas))
        stats.valid_signal_frames = int(sum(count > 0 for count in valid_counts))
        if geometry_areas:
            areas = np.asarray(geometry_areas, dtype=float)
            valid = np.asarray(valid_counts, dtype=float)
            stats.roi_area_mean = float(np.mean(areas))
            stats.roi_area_std = float(np.std(areas))
            stats.signal_pixel_count_mean = float(np.mean(valid))
            stats.signal_pixel_count_std = float(np.std(valid))
            ratios = np.divide(valid, areas, out=np.zeros_like(valid), where=areas > 0)
            stats.valid_pixel_ratio_mean = float(np.mean(ratios))
            stats.valid_pixel_ratio_std = float(np.std(ratios))
            total_area = float(np.sum(areas))
            if total_area > 0:
                stats.rejected_dark_pixel_ratio = float(np.sum(rejected_dark_counts) / total_area)
                stats.rejected_bright_pixel_ratio = float(np.sum(rejected_bright_counts) / total_area)

        if geometry_frames:
            pixels = np.concatenate(geometry_frames, axis=0)
            if pixels.size:
                intensity = pixels.mean(axis=1)
                luma = encoded_luma(pixels)
                stats.mean_rgb = float(np.mean(intensity))
                stats.std_rgb = float(np.std(intensity))
                stats.p95_rgb = float(np.percentile(intensity, 95))
                stats.p99_rgb = float(np.percentile(intensity, 99))
                stats.sat_ratio_250 = float(np.mean(np.any(pixels >= 250, axis=1)))
                stats.sat_ratio_245 = float(np.mean(np.any(pixels >= 245, axis=1)))
                stats.dark_ratio_5 = float(np.mean(np.any(pixels <= 5, axis=1)))
                stats.dark_ratio_10 = float(np.mean(np.any(pixels <= 10, axis=1)))
                stats.dark_ratio_20 = float(np.mean(luma <= 20))
                stats.bright_ratio_220 = float(np.mean(luma >= 220))
                stats.bright_ratio_235 = float(np.mean(luma >= 235))

                means = pixels.mean(axis=0)
                medians = np.median(pixels, axis=0)
                stats.mean_r, stats.mean_g, stats.mean_b = map(float, means)
                stats.roi_mean_r, stats.roi_mean_g, stats.roi_mean_b = map(float, means)
                stats.roi_median_r, stats.roi_median_g, stats.roi_median_b = map(float, medians)
                eps = 1e-8
                stats.rg_ratio = stats.r_over_g = float(means[0] / (means[1] + eps))
                stats.bg_ratio = stats.b_over_g = float(means[2] / (means[1] + eps))
                stats.r_over_b = float(means[0] / (means[2] + eps))
                stats.r_minus_g = float(means[0] - means[1])
                stats.g_minus_b = float(means[1] - means[2])
                stats.channel_range = float(np.max(means) - np.min(means))

                stats.roi_luma_mean = float(np.mean(luma))
                stats.roi_luma_median = float(np.median(luma))
                stats.roi_luma_std = float(np.std(luma))
                p05, p25, p75, p95, p99 = np.percentile(luma, [5, 25, 75, 95, 99])
                stats.roi_luma_p05 = float(p05)
                stats.roi_luma_p25 = float(p25)
                stats.roi_luma_p75 = float(p75)
                stats.roi_luma_p95 = float(p95)
                stats.roi_luma_p99 = float(p99)
                stats.roi_luma_iqr = float(p75 - p25)
                stats.roi_luma_p95_minus_p05 = float(p95 - p05)

        if frame_rgb_means:
            frame_means = np.asarray(frame_rgb_means, dtype=float)
            if frame_means.shape[0] >= 1:
                stds = np.std(frame_means, axis=0)
                means = np.mean(frame_means, axis=0)
                cvs = stds / (means + 1e-8)
                stats.temporal_r_std, stats.temporal_g_std, stats.temporal_b_std = map(float, stds)
                stats.temporal_r_cv, stats.temporal_g_cv, stats.temporal_b_cv = map(float, cvs)
                stats.raw_cv_r, stats.raw_cv_g, stats.raw_cv_b = map(float, cvs)

        if frame_luma_means:
            temporal = np.asarray(frame_luma_means, dtype=float)
            stats.temporal_luma_std = float(np.std(temporal))
            stats.temporal_luma_cv = float(np.std(temporal) / (np.mean(temporal) + 1e-8))
            if temporal.size >= 2:
                changes = np.abs(np.diff(temporal))
                stats.mean_frame_to_frame_luma_change = float(np.mean(changes))
                stats.p95_frame_to_frame_luma_change = float(np.percentile(changes, 95))

        if spatial_rows:
            for field in (
                "spatial_luma_std", "spatial_luma_iqr", "highlight_component_ratio",
                "shadow_component_ratio", "left_right_luma_difference",
                "upper_lower_luma_difference",
            ):
                setattr(stats, field, cls._safe_mean([row.get(field, np.nan) for row in spatial_rows]))

        if valid_frames:
            valid_pixels = np.concatenate(valid_frames, axis=0)
            if valid_pixels.size:
                valid_intensity = valid_pixels.mean(axis=1)
                valid_luma = encoded_luma(valid_pixels)
                stats.valid_mean_rgb = float(np.mean(valid_intensity))
                stats.valid_std_rgb = float(np.std(valid_intensity))
                valid_means = valid_pixels.mean(axis=0)
                stats.valid_mean_r, stats.valid_mean_g, stats.valid_mean_b = map(float, valid_means)
                stats.valid_pixel_luma_mean = float(np.mean(valid_luma))
                stats.valid_pixel_luma_std = float(np.std(valid_luma))
        return stats

    @staticmethod
    def _labeled_panel(rgb: np.ndarray, label: str, target_height: int) -> np.ndarray:
        image = rgb.copy()
        if image.shape[0] != target_height:
            width = max(1, int(image.shape[1] * target_height / image.shape[0]))
            image = cv2.resize(image, (width, target_height))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (min(280, bgr.shape[1]), 32), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return bgr

    def _save_example(self, original_rgb: np.ndarray, gamma_rgb: np.ndarray, full_skin_rgb: np.ndarray) -> None:
        if self.example_output_path is None or self._example_saved:
            return
        panels = [
            self._labeled_panel(original_rgb, "Original RGB", 360),
            self._labeled_panel(gamma_rgb, f"Gamma={self.gamma_value:g}", 360),
            self._labeled_panel(full_skin_rgb, "Gamma-frame skin ROI", 360),
        ]
        montage = cv2.hconcat(panels)
        self.example_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.example_output_path), montage)
        self._example_saved = True

    def extract_holistic(self, videoFileName):
        signal, stats = self.extract_holistic_with_metrics(videoFileName)
        self.last_brightness_stats = stats
        return signal

    def extract_holistic_with_metrics(self, videoFileName) -> Tuple[np.ndarray, Dict[str, float]]:
        self.visualize_skin_collection = []
        skin_extractor = self.skin_extractor
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        presence_threshold = 0.5
        visibility_threshold = 0.5

        signal = []
        processed_frames = 0
        face_detected_frames = 0
        geometry_frames: List[np.ndarray] = []
        valid_frames: List[np.ndarray] = []
        geometry_areas: List[int] = []
        valid_counts: List[int] = []
        rejected_dark_counts: List[int] = []
        rejected_bright_counts: List[int] = []
        frame_rgb_means: List[np.ndarray] = []
        frame_luma_means: List[float] = []
        spatial_rows: List[Dict[str, float]] = []

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            for frame in iter_frames_inline(videoFileName):
                original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gamma_rgb = self._apply_gamma(original_rgb)
                detection_rgb = gamma_rgb
                signal_rgb = gamma_rgb

                processed_frames += 1
                height, width = detection_rgb.shape[:2]
                landmarks_array = np.zeros((468, 5), dtype=np.float32)
                landmarks_array[:, :2] = -1.0
                results = face_mesh.process(detection_rgb)

                cropped_mask_im = None
                if results.multi_face_landmarks:
                    face_detected_frames += 1
                    for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                        invisible = (
                            landmark.HasField("visibility") and landmark.visibility < visibility_threshold
                        ) or (
                            landmark.HasField("presence") and landmark.presence < presence_threshold
                        )
                        if invisible:
                            continue
                        coords = mp_drawing._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, width, height
                        )
                        if coords:
                            landmarks_array[idx, 0] = coords[1]
                            landmarks_array[idx, 1] = coords[0]
                    cropped_skin_im, full_skin_im = skin_extractor.extract_skin(signal_rgb, landmarks_array)
                    if self.collect_brightness:
                        white = np.full_like(signal_rgb, 255, dtype=np.uint8)
                        cropped_mask_im, _ = skin_extractor.extract_skin(white, landmarks_array)
                    self._save_example(original_rgb, gamma_rgb, full_skin_im)
                else:
                    cropped_skin_im = np.zeros_like(signal_rgb)
                    full_skin_im = np.zeros_like(signal_rgb)

                if self.visualize_skin:
                    self.visualize_skin_collection.append(full_skin_im)

                signal.append(holistic_mean(
                    cropped_skin_im,
                    np.int32(SignalProcessingParams.RGB_LOW_TH),
                    np.int32(SignalProcessingParams.RGB_HIGH_TH),
                ))

                if self.collect_brightness:
                    geometry = self._geometry_pixels(cropped_skin_im, cropped_mask_im)
                    valid = self._signal_valid_pixels(geometry)
                    geometry_areas.append(int(geometry.shape[0]))
                    valid_counts.append(int(valid.shape[0]))
                    lo = int(SignalProcessingParams.RGB_LOW_TH)
                    hi = int(SignalProcessingParams.RGB_HIGH_TH)
                    rejected_dark_counts.append(int(np.sum(np.any(geometry < lo, axis=1))) if geometry.size else 0)
                    rejected_bright_counts.append(int(np.sum(np.any(geometry > hi, axis=1))) if geometry.size else 0)
                    if geometry.shape[0] > 0:
                        bounded = self._bounded_sample(geometry)
                        geometry_frames.append(bounded)
                        frame_rgb_means.append(geometry.mean(axis=0))
                        frame_luma_means.append(float(np.mean(encoded_luma(geometry))))
                        spatial_rows.append(self._spatial_metrics(cropped_skin_im, cropped_mask_im))
                    if valid.shape[0] > 0:
                        valid_frames.append(self._bounded_sample(valid))

                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames >= self.tot_frames:
                    break

        signal_array = np.asarray(signal, dtype=np.float32)
        stats = self._summarize_brightness(
            geometry_frames, valid_frames, geometry_areas, valid_counts, face_detected_frames,
            frame_rgb_means, frame_luma_means, spatial_rows,
            rejected_dark_counts, rejected_bright_counts,
        ).to_dict()
        self.last_brightness_stats = stats
        return signal_array, stats
