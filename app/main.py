import os
import json

from app.utils.image_io import load_image
from app.clients.gemma_client import GemmaClient
from app.services.step1_extract import MenuExtractor
from app.services.step2_rank import MenuRanker

#MY_KEY = "AIzaSyC7Zgp9nWiFTkweTy2gzNXx-8xbogIxYHQ"
gemma = GemmaClient(api_key=os.getenv("GOOGLE_API_KEY"))
extractor = MenuExtractor(gemma)
ranker = MenuRanker(gemma, uncertainty_penalty=60)

data, mime = load_image("menu.png")
img = gemma.image_part_from_bytes(data, mime)  # py 파일과 같은 폴더면 OK

# data, mime = load_image("https://example.com/menu.png")
# img_part = gemma.image_part_from_bytes(data, mime)

# step 1. Extract dish name from menu
items = extractor.extract(img)
#print(items)

# step 2. Scoring and Ranking to each dish
avoid = ["계란", "돼지고기", "땅콩", "고수"]

result = ranker.rank(items, avoid)
#print(json.dumps(result, ensure_ascii=False, indent=2))