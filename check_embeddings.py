import os
import numpy as np
from numpy.linalg import norm

EMBEDDINGS_DIR = "data/embeddings"

users = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith("_embeddings.npy")]
if not users:
    print("임베딩 파일이 없습니다. 단계 5를 먼저 실행하세요.")
    exit()

emb_dict = {}

print("=== 임베딩 파일 확인 및 shape ===")
for f in users:
    path = os.path.join(EMBEDDINGS_DIR, f)
    emb = np.load(path)
    emb_dict[f] = emb
    print(f"{f}: shape={emb.shape}, min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")

print("\n=== 사용자 간 거리 확인 (샘플) ===")
user_files = list(emb_dict.keys())
for i in range(len(user_files)):
    for j in range(i+1, len(user_files)):
        u1, u2 = user_files[i], user_files[j]
        # 첫 벡터 기준 거리
        dist = norm(emb_dict[u1][0] - emb_dict[u2][0])
        print(f"Distance between {u1} and {u2}: {dist:.4f}")
