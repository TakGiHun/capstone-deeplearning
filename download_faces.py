import os
import requests
from tqdm import tqdm

# ======================================
# 설정값
# ======================================
NUM_USERS = 3        # 생성할 가짜 사용자 수
IMAGES_PER_USER = 5  # 각 사용자당 이미지 수
SAVE_DIR = "data"    # 저장 경로 (자동 생성됨)
URL = "https://thispersondoesnotexist.com/"  # 이미지 생성 사이트
# ======================================

def download_faces():
    os.makedirs(SAVE_DIR, exist_ok=True)
    total = NUM_USERS * IMAGES_PER_USER

    print(f"📸 총 {total}장의 합성 얼굴 이미지를 다운로드합니다...\n")

    for user_id in range(1, NUM_USERS + 1):
        user_dir = os.path.join(SAVE_DIR, f"user{user_id:02d}")
        os.makedirs(user_dir, exist_ok=True)

        for img_id in tqdm(range(1, IMAGES_PER_USER + 1), desc=f"user{user_id:02d}"):
            try:
                response = requests.get(URL, timeout=10)
                img_path = os.path.join(user_dir, f"user{user_id:02d}_{img_id:02d}.jpg")
                with open(img_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"❌ 다운로드 실패: {e}")

    print("\n✅ 다운로드 완료!")
    print(f"📂 저장 경로: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    download_faces()
