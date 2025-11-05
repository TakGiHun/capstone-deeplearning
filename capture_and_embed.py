import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# GPU 사용 가능하면 GPU, 없으면 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN과 ResNet 초기화
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 임베딩 저장 폴더
EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# 사용자 이름 입력
user_name = input("등록할 사용자 이름 입력: ").strip()
user_file = os.path.join(EMBEDDINGS_DIR, f"{user_name}_embeddings.npy")

# 웹캠 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠 열기 실패")
    exit()

print("웹캠에서 얼굴 5장 촬영 후 임베딩 생성")
print("촬영 준비 완료되면 's' 키 누르세요")

embeddings = []

while len(embeddings) < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    # 얼굴이 한 명만 검출될 때만 처리
    if boxes is not None and len(boxes) == 1:
        x1, y1, x2, y2 = map(int, boxes[0])
        face = img_rgb[y1:y2, x1:x2]

        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Captured: {len(embeddings)}/5", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and boxes is not None and len(boxes) == 1:
        # Tensor 변환
        face_tensor = torch.tensor(face).permute(2,0,1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        # 임베딩 추출
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()[0]
            embeddings.append(embedding)
            print(f"{len(embeddings)}/5 임베딩 저장 완료")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if embeddings:
    embeddings = np.stack(embeddings)
    np.save(user_file, embeddings)
    print(f"{user_name} 임베딩 파일 저장 완료: {user_file}")
else:
    print("임베딩 생성 실패")
