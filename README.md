---
title: Gdg Team 9
emoji: 💻
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local Environment Variables

1. Copy `.env.example` to `.env`.
2. Fill at least `GOOGLE_API_KEY`.
3. If you want a separate Vision key for bbox OCR, set `GOOGLE_VISION_API_KEY`.

```bash
cp .env.example .env
```

`.env` example:

```env
GOOGLE_API_KEY=your-google-api-key
GOOGLE_VISION_API_KEY=your-google-vision-key
```

Notes:
- `GOOGLE_VISION_API_KEY` is optional. If not set, bbox Vision API uses `GOOGLE_API_KEY`.
- `.env` is auto-loaded by `app/api.py`, `app/main.py`, and `run_full_cycle_experiment.py`.

## OCR Settings (Vision)

This project now uses Google Vision OCR only.

Optional `.env`:

```env
OCR_VISION_TIMEOUT_SEC=25
```

## Quick Local Test

```bash
python run_full_cycle_experiment.py \
  --image menu_image/menu0.jpeg \
  --avoid peanuts cheese \
  --presigned-url "<S3_PRESIGNED_PUT_URL>"
```

## OCR Only Test (Google Vision)

```bash
python run_ocr_only_experiment.py \
  --image menu_image/menu_korean.png \
  --with-preprocess \
  --menu-country-code AUTO
```
