# run_preprocess_local.py
from app.agents.preprocess_agent import ImagePreprocessAgent
from app.utils.image_io import load_image

# 입력 이미지 (로컬 파일)
src = "menu.png"   # 또는 "menu.png"
# 저장 경로
dst = "debug/preprocessed_menu.png"

data, mime = load_image(src)

agent = ImagePreprocessAgent(
    min_short_edge=900,
)

out_data, out_mime = agent.run(data, mime, save_path=dst)

print("done")
print("source:", src)
print("saved :", dst)
print("mime  :", out_mime)
print("bytes :", len(out_data))
