"""Multi-method OCR pipeline based on DeepSeek OCR research."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Callable


@dataclass
class BoundingBox:
    """Bounding box for detected text."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0


@dataclass
class OCRResult:
    """Result from OCR extraction."""
    text: str
    confidence: float
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    method: str = "unknown"
    
    @property
    def normalized_text(self) -> str:
        """Return lowercase, whitespace-normalized text."""
        import re
        return re.sub(r"\s+", " ", self.text).strip().lower()


@dataclass
class UIElement:
    """Detected UI element based on DinoV3 concepts."""
    element_type: str  # button, input, modal, link, dropdown, checkbox
    label: str | None
    state: str  # visible, hidden, disabled, focused, selected
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float


class OCRPipeline:
    """
    Multi-method OCR pipeline that chains different OCR approaches.
    
    Based on DeepSeek OCR research:
    - Chain multiple methods for robustness
    - Use confidence scoring to pick best result
    - Fall back to simpler methods if advanced ones fail
    """
    
    def __init__(
        self,
        methods: list[str] | None = None,
        min_confidence: float = 0.3,
    ):
        self.methods = methods or ["tesseract"]
        self.min_confidence = min_confidence
        self._method_handlers: dict[str, Callable] = {
            "tesseract": self._run_tesseract,
            "tesseract_best": self._run_tesseract_best,
        }
    
    def extract_text(self, image_path: Path) -> OCRResult:
        """
        Extract text from image using chained OCR methods.
        
        Returns the highest confidence result that meets minimum threshold.
        """
        if not image_path.exists():
            return OCRResult(text="", confidence=0.0, method="none")
        
        results: list[OCRResult] = []
        
        for method in self.methods:
            handler = self._method_handlers.get(method)
            if handler:
                try:
                    result = handler(image_path)
                    if result.confidence >= self.min_confidence:
                        return result  # Return first good result
                    results.append(result)
                except Exception:
                    continue
        
        # Return best result if none meet threshold
        if results:
            return max(results, key=lambda r: r.confidence)
        
        return OCRResult(text="", confidence=0.0, method="none")
    
    def _run_tesseract(self, image_path: Path) -> OCRResult:
        """Run Tesseract OCR with standard settings."""
        tesseract = which("tesseract")
        if not tesseract:
            return OCRResult(text="", confidence=0.0, method="tesseract")
        
        cmd = [tesseract, str(image_path), "stdout", "-l", "eng"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            return OCRResult(text="", confidence=0.0, method="tesseract")
        
        text = result.stdout.strip()
        # Estimate confidence based on text quality
        confidence = self._estimate_confidence(text)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            method="tesseract",
        )
    
    def _run_tesseract_best(self, image_path: Path) -> OCRResult:
        """Run Tesseract with best accuracy settings (slower)."""
        tesseract = which("tesseract")
        if not tesseract:
            return OCRResult(text="", confidence=0.0, method="tesseract_best")
        
        cmd = [
            tesseract,
            str(image_path),
            "stdout",
            "-l", "eng",
            "--oem", "1",  # LSTM only
            "--psm", "3",  # Fully automatic page segmentation
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            return OCRResult(text="", confidence=0.0, method="tesseract_best")
        
        text = result.stdout.strip()
        confidence = self._estimate_confidence(text) * 1.1  # Slight boost for best mode
        
        return OCRResult(
            text=text,
            confidence=min(1.0, confidence),
            method="tesseract_best",
        )
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate OCR confidence based on text characteristics.
        
        Heuristics:
        - Ratio of alphanumeric to garbage characters
        - Presence of common words
        - Average word length
        """
        if not text:
            return 0.0
        
        # Character quality ratio
        alpha_num = sum(1 for c in text if c.isalnum() or c.isspace())
        total = len(text)
        char_ratio = alpha_num / total if total > 0 else 0
        
        # Word quality
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        word_score = min(1.0, avg_word_len / 8)  # Normalize to ~8 char avg
        
        # Common UI keywords
        ui_keywords = [
            "button", "click", "submit", "search", "login", "cart",
            "menu", "home", "add", "view", "price", "buy", "shop",
        ]
        keyword_hits = sum(1 for kw in ui_keywords if kw in text.lower())
        keyword_score = min(1.0, keyword_hits * 0.15)
        
        # Combined confidence
        confidence = (char_ratio * 0.5) + (word_score * 0.3) + (keyword_score * 0.2)
        return min(1.0, max(0.0, confidence))
    
    def extract_with_boxes(self, image_path: Path) -> OCRResult:
        """
        Extract text with bounding boxes for element localization.
        
        Uses Tesseract's TSV output for box coordinates.
        """
        tesseract = which("tesseract")
        if not tesseract or not image_path.exists():
            return OCRResult(text="", confidence=0.0, method="tesseract_boxes")
        
        cmd = [
            tesseract,
            str(image_path),
            "stdout",
            "-l", "eng",
            "--oem", "1",
            "-c", "tessedit_create_tsv=1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            return OCRResult(text="", confidence=0.0, method="tesseract_boxes")
        
        boxes = []
        full_text_parts = []
        
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split("\t")
            if len(parts) >= 12:
                text = parts[11]
                if text.strip():
                    try:
                        boxes.append(BoundingBox(
                            text=text,
                            x=int(parts[6]),
                            y=int(parts[7]),
                            width=int(parts[8]),
                            height=int(parts[9]),
                            confidence=float(parts[10]) / 100.0,
                        ))
                        full_text_parts.append(text)
                    except (ValueError, IndexError):
                        continue
        
        full_text = " ".join(full_text_parts)
        avg_confidence = (
            sum(b.confidence for b in boxes) / len(boxes)
            if boxes else 0.0
        )
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=boxes,
            method="tesseract_boxes",
        )


class FrameEnhancer:
    """
    Enhance video frames for better OCR based on Restormer concepts.
    
    Uses PIL/Pillow for image processing:
    - Denoising
    - Sharpening
    - Contrast enhancement
    """
    
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir
    
    def enhance(self, frame_path: Path) -> Path:
        """
        Enhance a frame for better OCR.
        
        Returns path to enhanced frame (may be same as input if no changes).
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter
        except ImportError:
            return frame_path  # Return original if Pillow not available
        
        if not frame_path.exists():
            return frame_path
        
        try:
            img = Image.open(frame_path)
            
            # 1. Slight sharpening
            img = img.filter(ImageFilter.SHARPEN)
            
            # 2. Contrast enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # 20% more contrast
            
            # 3. Brightness adjustment (slight)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)  # 5% brighter
            
            # Save enhanced frame
            if self.output_dir:
                output_path = self.output_dir / f"enhanced_{frame_path.name}"
            else:
                output_path = frame_path.parent / f"enhanced_{frame_path.name}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            
            return output_path
        
        except Exception:
            return frame_path  # Return original on any error
    
    def enhance_batch(self, frame_paths: list[Path]) -> list[Path]:
        """Enhance multiple frames."""
        return [self.enhance(p) for p in frame_paths]


def create_ocr_pipeline(methods: list[str] | None = None) -> OCRPipeline:
    """Factory function to create an OCR pipeline."""
    return OCRPipeline(methods=methods)


def create_frame_enhancer(output_dir: Path | None = None) -> FrameEnhancer:
    """Factory function to create a frame enhancer."""
    return FrameEnhancer(output_dir=output_dir)
