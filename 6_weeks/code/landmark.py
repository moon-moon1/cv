import cv2 as cv                                              # OpenCV를 불러옴
import mediapipe as mp                                        # MediaPipe를 불러옴

# MediaPipe FaceMesh 모듈을 가져옴
mp_face_mesh = mp.solutions.face_mesh

# 확인할 사진 파일을 읽어옴
img = cv.imread("face.jpg")

# 이미지가 정상적으로 읽혔는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 원본 이미지를 복사해서 결과를 그릴 이미지로 사용
output = img.copy()

# 이미지 크기를 구함
h, w, _ = img.shape

# OpenCV는 BGR 형식이므로 MediaPipe 입력용 RGB로 변환
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 정적 이미지용 FaceMesh 객체 생성
# refine_landmarks=False 이면 기본 468개 랜드마크를 사용
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
) as face_mesh:

    # 사진 1장을 넣어서 얼굴 랜드마크를 검출
    results = face_mesh.process(rgb)

    # 얼굴 랜드마크가 검출되었는지 확인
    if results.multi_face_landmarks:
        # 검출된 얼굴마다 반복
        for face_landmarks in results.multi_face_landmarks:
            # 468개 랜드마크 각각에 대해 반복
            for landmark in face_landmarks.landmark:
                # 정규화 좌표를 실제 픽셀 좌표로 변환
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # 점으로 시각화
                cv.circle(output, (x, y), 1, (0, 255, 0), -1)

        print("얼굴 랜드마크 검출 완료")
    else:
        print("얼굴을 찾지 못했습니다.")

# 결과 이미지를 저장
cv.imwrite("face_landmarks_result.jpg", output)

# 결과 이미지를 화면에 표시
cv.imshow("Face Landmarks", output)

# 아무 키나 누를 때까지 대기
cv.waitKey(0)

# 창 닫기
cv.destroyAllWindows()