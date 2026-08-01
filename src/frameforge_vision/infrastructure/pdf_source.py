"""Document source adapter: rasterise a PDF page (PyMuPDF) into a RasterImage.

Reuses the repo's existing ``pdf`` dependency group (PyMuPDF / ``fitz``). When
PyMuPDF is absent the source reports ``available() is False`` so the document
tool fails with a clear, actionable message instead of an import traceback.
"""
from __future__ import annotations

from ..domain.observation import RasterImage


class PdfDocumentSource:
    name = "pdf"

    def available(self) -> bool:
        try:
            import fitz  # noqa: F401  (PyMuPDF)
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return "PDF rasterisation requires PyMuPDF; install the `pdf` dependency group"

    def render_page(self, path: str, page: int, *, dpi: int = 144) -> RasterImage:
        import fitz

        if page < 1:
            raise ValueError("page is 1-based and must be >= 1")
        with fitz.open(path) as document:
            if page > document.page_count:
                raise ValueError(f"page {page} out of range (document has {document.page_count})")
            pdf_page = document.load_page(page - 1)
            pixmap = pdf_page.get_pixmap(dpi=dpi, alpha=False)
            png_bytes = pixmap.tobytes("png")

        pixels = self._pixels(pixmap)
        return RasterImage(
            width=int(pixmap.width),
            height=int(pixmap.height),
            pixels=pixels,
            encoded=png_bytes,
            media_type="image/png",
        )

    @staticmethod
    def _pixels(pixmap):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - exercised only without numpy
            return None
        arr = np.frombuffer(pixmap.samples, dtype="uint8")
        arr = arr.reshape(pixmap.height, pixmap.width, pixmap.n)
        return np.ascontiguousarray(arr[:, :, :3])
