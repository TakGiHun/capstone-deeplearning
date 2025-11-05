# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, os
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 환경 / 모델 초기화 (서버 시작 시 한 번만)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device, min_face_size=80)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 임베딩 로드
EMB_DIR = "data/embeddings"
emb_dict = {}
for f in os.listdir(EMB_DIR):
    if f.endswith("_embeddings.npy"):
        emb_dict[f.replace("_embeddings.npy","")] = np.load(os.path.join(EMB_DIR, f))

THRESHOLD = 0.9

def pil_from_bytes(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

@app.post("/api/face-auth")
async def face_auth(file: UploadFile = File(...)):
    contents = await file.read()
    img = pil_from_bytes(contents)

    # MTCNN이 정렬된 텐서 반환 (None 또는 Tensor)
    face_tensor = mtcnn(img)   # (3,160,160) or None
    if face_tensor is None:
        raise HTTPException(status_code=400, detail="No face detected")

    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)  # (1,3,160,160)

    with torch.no_grad():
        emb = resnet(face_tensor.to(device)).cpu().numpy()[0]

    best_user, best_dist = None, float('inf')
    for user, user_embs in emb_dict.items():
        for ue in user_embs:
            d = norm(emb - ue)
            if d < best_dist:
                best_dist = d
                best_user = user

    status = "accepted" if best_dist < THRESHOLD else "unknown"
    return {"status": status, "user": best_user if status=="accepted" else None, "distance": float(best_dist)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
