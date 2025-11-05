import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np  # 색상 변환용

# GPU 사용 가능하면 GPU, 없으면 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# 원본 데이터 폴더
DATA_DIR = "data"              # 다운로드한 얼굴 이미지 폴더
ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
os.makedirs(ALIGNED_DIR, exist_ok=True)

# user 폴더 목록
users = [d for d in os.listdir(DATA_DIR) 
         if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("user")]

for user in users:
    user_dir = os.path.join(DATA_DIR, user)
    aligned_user_dir = os.path.join(ALIGNED_DIR, user)
    os.makedirs(aligned_user_dir, exist_ok=True)

    images = [f for f in os.listdir(user_dir) if f.lower().endswith(".jpg")]
    
    for img_name in tqdm(images, desc=f"Processing {user}"):
        img_path = os.path.join(user_dir, img_name)
        img = Image.open(img_path)

        # 얼굴 검출 & 정렬
        face_tensor = mtcnn(img)
        if face_tensor is not None:
            # CPU로 이동
            face_tensor = face_tensor.cpu()

            # [C,H,W] -> [H,W,C], 0~255 범위, uint8로 변환
            face_np = (face_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)

            # PIL 이미지로 변환 후 저장
            face_pil = Image.fromarray(face_np)
            save_path = os.path.join(aligned_user_dir, img_name)
            face_pil.save(save_path)
