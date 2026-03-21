import base64
import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests


class MenuBBoxHighlighterAgent:
    """
    상위 메뉴명을 Google Vision OCR paragraph 텍스트와 매칭해
    bounding box를 그린 뒤 업로드하고 결과 URL을 반환한다.
    - 1순위: 요청에서 전달받은 presigned PUT URL로 업로드
    - 2순위: 기존 내부 S3 업로드 + presigned GET 생성(옵션)
    """

    def __init__(
        self,
        vision_api_key: str | None = None,
        s3_bucket: str | None = None,
        s3_region: str | None = None,
        s3_prefix: str | None = None,
        local_output_dir: str | None = None,
        presigned_expires_sec: int | None = None,
        request_type: str = "DOCUMENT_TEXT_DETECTION",
    ):
        self.vision_api_key = (vision_api_key or os.getenv("GOOGLE_VISION_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.s3_bucket = (s3_bucket or os.getenv("BBOX_S3_BUCKET") or "").strip()
        self.s3_region = (s3_region or os.getenv("BBOX_S3_REGION") or "").strip()
        self.s3_prefix = (s3_prefix or os.getenv("BBOX_S3_PREFIX") or "menu-bbox").strip().strip("/")
        self.local_output_dir = (local_output_dir or os.getenv("BBOX_LOCAL_OUTPUT_DIR") or "debug/bbox").strip()
        self.presigned_expires_sec = max(
            60,
            int((presigned_expires_sec or int(os.getenv("BBOX_PRESIGNED_EXPIRES_SEC", "3600")))),
        )
        self.request_type = (request_type or "DOCUMENT_TEXT_DETECTION").strip().upper()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).strip().casefold()

    @staticmethod
    def _normalize_targets(targets: Sequence[str], limit: int = 3) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in targets:
            cleaned = re.sub(r"\s+", " ", str(item or "")).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _predict(self, image_bytes: bytes):
        content = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.vision_api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": self.request_type}],
                }
            ]
        }
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _paragraph_text(paragraph: dict) -> str:
        words = paragraph.get("words", []) or []
        return "".join(
            "".join(str(symbol.get("text", "")) for symbol in (word.get("symbols", []) or []))
            for word in words
        )

    @staticmethod
    def _polygon_from_vertices(vertices: Sequence[dict]) -> Tuple[Tuple[int, int], ...] | None:
        points: List[Tuple[int, int]] = []
        for vertex in vertices or []:
            if not isinstance(vertex, dict):
                continue
            x = int(vertex.get("x", 0) or 0)
            y = int(vertex.get("y", 0) or 0)
            points.append((x, y))
        if len(points) < 3:
            return None
        return tuple(points)

    def _collect_target_boxes(self, data: dict, targets: Sequence[str]) -> Tuple[List[Tuple[Tuple[int, int], ...]], List[str]]:
        normalized_targets = [(target, self._normalize_text(target)) for target in targets]
        normalized_targets = [(raw, norm) for raw, norm in normalized_targets if norm]
        if not normalized_targets:
            return [], []

        responses = data.get("responses", []) if isinstance(data, dict) else []
        if not responses:
            return [], []
        annotation = responses[0].get("fullTextAnnotation", {}) if isinstance(responses[0], dict) else {}
        pages = annotation.get("pages", []) if isinstance(annotation, dict) else []

        polygons: List[Tuple[Tuple[int, int], ...]] = []
        polygon_seen = set()
        matched_targets: List[str] = []
        matched_seen = set()

        for page in pages:
            for block in page.get("blocks", []) or []:
                for paragraph in block.get("paragraphs", []) or []:
                    paragraph_text = self._normalize_text(self._paragraph_text(paragraph))
                    if not paragraph_text:
                        continue

                    matched_here = False
                    for raw_target, norm_target in normalized_targets:
                        if norm_target in paragraph_text:
                            matched_here = True
                            if raw_target.casefold() not in matched_seen:
                                matched_seen.add(raw_target.casefold())
                                matched_targets.append(raw_target)

                    if not matched_here:
                        continue

                    polygon = self._polygon_from_vertices(
                        (paragraph.get("boundingBox", {}) or {}).get("vertices", []) if isinstance(paragraph, dict) else []
                    )
                    if not polygon:
                        continue
                    if polygon in polygon_seen:
                        continue
                    polygon_seen.add(polygon)
                    polygons.append(polygon)

        return polygons, matched_targets

    @staticmethod
    def _draw_boxes(image_bytes: bytes, polygons: Sequence[Tuple[Tuple[int, int], ...]]) -> bytes:
        from PIL import Image, ImageDraw

        image = Image.open(io.BytesIO(image_bytes))
        drawer = ImageDraw.Draw(image)
        for polygon in polygons:
            drawer.polygon(list(polygon), outline="green", width=3)

        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    @staticmethod
    def _public_url_from_presigned(upload_presigned_url: str) -> str:
        if not upload_presigned_url:
            return ""
        parts = urlsplit(upload_presigned_url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def _upload_to_presigned_url(upload_presigned_url: str, image_bytes: bytes) -> str:
        """
        S3 presigned PUT URL에 파일 바이트를 그대로 업로드한다.
        성공 시 query를 제거한 object URL(path)을 반환한다.
        """
        if not upload_presigned_url:
            return ""
        response = requests.put(upload_presigned_url, data=image_bytes, timeout=20)
        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(f"presigned upload failed: status={response.status_code}, body={response.text[:200]}")
        return MenuBBoxHighlighterAgent._public_url_from_presigned(upload_presigned_url)

    def _upload_and_presign(self, image_bytes: bytes) -> str:
        if not self.s3_bucket:
            return ""
        try:
            import boto3
        except Exception:
            return ""

        client_kwargs = {}
        if self.s3_region:
            client_kwargs["region_name"] = self.s3_region
        s3 = boto3.client("s3", **client_kwargs)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"{self.s3_prefix}/{timestamp}-{uuid4().hex}.png" if self.s3_prefix else f"{timestamp}-{uuid4().hex}.png"

        s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
        )
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.s3_bucket, "Key": key},
            ExpiresIn=self.presigned_expires_sec,
        )

    def _save_local(self, image_bytes: bytes) -> str:
        if not self.local_output_dir:
            return ""
        directory = Path(self.local_output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"bbox-top3-{timestamp}-{uuid4().hex}.png"
        path.write_bytes(image_bytes)
        return str(path.resolve())

    def run(
        self,
        image_bytes: bytes,
        targets: Sequence[str],
        top_k: int = 3,
        upload_presigned_url: str = "",
    ) -> Tuple[str, List[str], str, int]:
        normalized_targets = self._normalize_targets(targets, limit=top_k)
        if not image_bytes or not normalized_targets:
            return "", normalized_targets, "", 0

        # 기본값은 원본 업로드. bbox를 찾으면 박스가 그려진 이미지로 대체한다.
        output_image = image_bytes
        matched_targets: List[str] = []
        if self.vision_api_key:
            data = self._predict(image_bytes)
            polygons, matched_targets = self._collect_target_boxes(data, normalized_targets)
            if polygons:
                output_image = self._draw_boxes(image_bytes, polygons)

        local_image_path = self._save_local(output_image)
        uploaded_url = ""
        upload_elapsed_ms = 0
        t_upload = time.perf_counter()
        if upload_presigned_url:
            uploaded_url = self._upload_to_presigned_url(upload_presigned_url, output_image)
        else:
            uploaded_url = self._upload_and_presign(output_image)
        upload_elapsed_ms = int(max(0, round((time.perf_counter() - t_upload) * 1000)))
        return uploaded_url, matched_targets, local_image_path, upload_elapsed_ms
