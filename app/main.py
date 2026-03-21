import os
import json

from app.agents._0_contracts import OCROptions
from app.agents._eval_2_ocr import OCRAgent
from app.agents._eval_3_extractor import OCRMenuJudgeAgent
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.utils.image_io import load_image
from app.utils.menu_item_cleaner import clean_menu_candidates
from app.utils.env_loader import load_local_env
from app.clients.gemma_client import GemmaClient
from app.services.step2_rank import MenuRanker

load_local_env()

gemma = GemmaClient(api_key=os.getenv("GOOGLE_API_KEY"))
ocr = OCRAgent(menu_country_code="AUTO")
extractor = OCRMenuJudgeAgent(gemma)
ranker = MenuRanker(gemma, uncertainty_penalty=60)
preprocess_agent = ImagePreprocessAgent()

data, mime = load_image("menu.png")
# 개발 중 전처리 결과를 확인할 수 있도록 로컬 파일로도 저장한다.
data, mime = preprocess_agent.run(data, mime, save_path="debug/preprocessed_menu.png")
ocr_out = ocr.run(data, options=OCROptions(min_confidence=0.5))

# step 1. Extract dish name from menu
judged = extractor.run_lines_with_image(
    lines=ocr_out.lines,
    image_bytes=data,
    image_mime=mime,
    use_image_context=True,
    ocr_lang=ocr_out.resolved_lang,
)
items = clean_menu_candidates(judged.menu_texts).cleaned_items

# step 2. Scoring and Ranking to each dish
avoid = ["계란", "돼지고기", "땅콩", "고수"]

result = ranker.rank(items, avoid)
print(json.dumps(result, ensure_ascii=False, indent=2))
