import cv2 as cv                                  # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import matplotlib.pyplot as plt                  # 결과 이미지를 출력하기 위해 matplotlib.pyplot을 plt로 불러옴

# mot_color70.jpg 파일을 컬러 이미지로 읽어서 img 변수에 저장
img = cv.imread('mot_color70.jpg')

# 이미지가 정상적으로 읽혔는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다.")          # 이미지를 읽지 못했을 때 오류 메시지 출력
    exit()                                        # 프로그램 종료

# OpenCV는 이미지를 BGR 형식으로 읽기 때문에
# matplotlib에서 올바른 색상으로 보기 위해 RGB 형식으로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# SIFT 특징점 검출은 보통 그레이스케일 이미지에서 수행하므로
# 컬러 이미지를 그레이스케일 이미지로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체를 생성
# nfeatures=200은 검출할 특징점 개수를 대략 제한하는 옵션
# 특징점이 너무 많으면 이 값을 줄이고, 더 많이 보고 싶으면 값을 늘리면 됨
sift = cv.SIFT_create(nfeatures=500)

# detectAndCompute()를 사용하여
# gray 이미지에서 특징점(keypoints)을 검출하고 기술자(descriptors)를 계산
# 첫 번째 반환값 keypoints는 특징점 정보 목록
# 두 번째 반환값 descriptors는 각 특징점을 수치 벡터로 표현한 기술자
keypoints, descriptors = sift.detectAndCompute(gray, None)

# drawKeypoints()를 사용하여 원본 이미지 위에 특징점을 그림
# None은 결과 이미지를 새로 생성하겠다는 의미
# flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 옵션은
# 특징점의 위치뿐 아니라 크기와 방향도 함께 시각화함
result = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# drawKeypoints() 결과도 OpenCV 형식(BGR)이므로
# matplotlib에 출력하기 위해 RGB 형식으로 변환
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# 전체 출력 창 크기를 설정
plt.figure(figsize=(14, 6))

# 1행 2열 중 첫 번째 위치에 원본 이미지를 출력
plt.subplot(1, 2, 1)

# RGB로 변환한 원본 이미지를 화면에 표시
plt.imshow(img_rgb)

# 첫 번째 이미지 제목을 설정
plt.title('Original Image')

# 축 눈금을 보이지 않게 설정
plt.axis('off')

# 1행 2열 중 두 번째 위치에 특징점 시각화 결과를 출력
plt.subplot(1, 2, 2)

# 특징점이 그려진 이미지를 화면에 표시
plt.imshow(result_rgb)

# 두 번째 이미지 제목을 설정
plt.title('SIFT Detect')

# 축 눈금을 보이지 않게 설정
plt.axis('off')

# subplot 간격을 자동으로 조정하여 보기 좋게 정리
plt.tight_layout()

# 최종 결과를 화면에 출력
plt.show()