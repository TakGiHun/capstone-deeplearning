import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
import os

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN 초기화
mtcnn = MTCNN(image_size=160, margin=20, device=device, min_face_size=80, keep_all=True)

# 얼굴 임베딩 모델
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 등록된 사용자 임베딩 불러오기
EMBEDDINGS_DIR = "data/embeddings"
emb_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith("_embeddings.npy")]
emb_dict = {f.replace("_embeddings.npy",""): np.load(os.path.join(EMBEDDINGS_DIR,f)) for f in emb_files}

# 인증 threshold
THRESHOLD = 0.9

# 카메라 자동 탐색
cap = None
for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        cap = test_cap
        print(f"사용 가능한 카메라 인덱스: {i}")
        break
    test_cap.release()

if cap is None:
    print("사용 가능한 카메라가 없습니다.")
    exit()

print("실시간 얼굴 인증 시작... 종료: 'q' 키")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # MTCNN에서 정렬된 얼굴 tensor 반환
        faces = mtcnn(img_rgb)  # 타입: None or Tensor (N,3,160,160)
        boxes, _ = mtcnn.detect(img_rgb)
    except Exception as e:
        print("MTCNN 에러:", e)
        faces, boxes = None, None

    identities = []
    if faces is not None and boxes is not None:
        if isinstance(faces, torch.Tensor):
            if faces.ndim == 3:
                face_tensors = faces.unsqueeze(0)
            else:
                face_tensors = faces
        else:
            face_tensors = None

        if face_tensors is not None:
            for idx in range(face_tensors.shape[0]):
                face_tensor = face_tensors[idx].unsqueeze(0).to(device)
                # 임베딩 추출
                try:
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu().numpy()[0]
                except RuntimeError as e:
                    print("모델 추론 에러(스킵):", e)
                    identities.append("Unknown")
                    continue

                # 사용자 임베딩과 거리 비교
                min_dist = float('inf')
                identity = "Unknown"
                for user, user_embs in emb_dict.items():
                    for uemb in user_embs:
                        d = norm(emb - uemb)
                        if d < min_dist:
                            min_dist = d
                            if d < THRESHOLD:
                                identity = user
                            else:
                                identity = "Unknown"
                identities.append(identity)

        # 얼굴마다 bbox 그리기
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            color = (0,255,0) if identities[idx] != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{identities[idx]}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
