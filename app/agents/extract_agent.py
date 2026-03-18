from typing import Optional

from google.genai import types

from app.agents._0_contracts import ExtractOutput
from app.agents._eval_2_ocr import OCRAgent
from app.agents._eval_3_extractor import OCRMenuJudgeAgent
from app.clients.gemma_client import GemmaClient


class MenuExtractAgent:
    """
    하위 호환용 추출 Agent.
    예전 `app.agents.extract_agent.MenuExtractAgent` 경로를 유지하면서
    현재 OCR + 메뉴 판독 조합으로 메뉴 후보를 추출한다.
    """

    def __init__(self, gemma: GemmaClient, menu_country_code: str = "KR"):
        self.ocr = OCRAgent(menu_country_code=menu_country_code)
        self.judge = OCRMenuJudgeAgent(gemma)

    def run(self, image_part: types.Part) -> ExtractOutput:
        image_bytes, image_mime = self._read_part_bytes(image_part)
        if not image_bytes:
            return ExtractOutput(items=[])
 
        ocr_out = self.ocr.run(image_bytes)
        judged = self.judge.run_lines_with_image(
            lines=ocr_out.lines,
            image_bytes=image_bytes,
            image_mime=image_mime or "image/jpeg",
            use_image_context=True,
        )
        return ExtractOutput(items=judged.menu_texts)

    @staticmethod
    def _read_part_bytes(image_part: types.Part) -> tuple[bytes, Optional[str]]:
        inline_data = getattr(image_part, "inline_data", None)
        if inline_data is None:
            return b"", None

        data = getattr(inline_data, "data", None)
        mime_type = getattr(inline_data, "mime_type", None)
        if isinstance(data, bytes):
            return data, mime_type

        if isinstance(data, str):
            try:
                return data.encode("utf-8"), mime_type
            except Exception:
                return b"", mime_type

        return b"", mime_type
