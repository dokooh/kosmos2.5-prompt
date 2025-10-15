"""
FP8 Mixed Precision Inference Package for KOSMOS-2.5

This package contains FP8 quantized inference modules for:
- OCR (Optical Character Recognition)
- Markdown generation
- Batch processing utilities

Modules:
- ocr_fp8_mixed: FP8 OCR inference engine
- md_fp8_mixed: FP8 Markdown inference engine
- batch_processor_fp8_mixed: Batch processing utilities
"""

from .ocr_fp8_mixed import EightBitOCRInference
from .md_fp8_mixed import EightBitMarkdownInference

__all__ = ['EightBitOCRInference', 'EightBitMarkdownInference']
__version__ = '1.0.0'