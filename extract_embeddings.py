import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

# GPU 사용 가능하면 GPU, 없으면 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pretrained 모델 로드 (VGGFace2 기준)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 정렬된 얼굴 이미지 폴더
ALIGNED_DIR = "data/aligned"
EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# 사용자별로 처리
users = [d for d in os.listdir(ALIGNED_DIR) if os.path.isdir(os.path.join(ALIGNED_DIR, d))]

for user in users:
    user_dir = os.path.join(ALIGNED_DIR, user)
    save_path = os.path.join(EMBEDDINGS_DIR, f"{user}_embeddings.npy")

    embeddings_list = []

    images = [f for f in os.listdir(user_dir) if f.lower().endswith(".jpg")]
    for img_name in tqdm(images, desc=f"Processing {user}"):
        img_path = os.path.join(user_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Tensor 변환
        img_tensor = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # 배치 차원 추가

        # 임베딩 추출
        with torch.no_grad():
            embedding = resnet(img_tensor)
        embeddings_list.append(embedding.cpu().numpy())

    # 사용자별 벡터 저장 (num_images × 512)
    embeddings_array = np.vstack(embeddings_list)
    np.save(save_path, embeddings_array)
