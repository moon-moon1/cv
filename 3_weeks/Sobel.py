import cv2 as cv                           # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import matplotlib.pyplot as plt           # 이미지 시각화를 위해 matplotlib의 pyplot을 plt로 불러옴
import numpy as np                        # 배열 및 수치 연산을 위해 numpy를 np로 불러옴

# 이미지를 컬러 형태로 읽어옴
img = cv.imread('edgeDetectionImage.jpg')

# 이미지가 정상적으로 불러와졌는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다. 파일명 또는 경로를 확인하세요.")
    exit()

# OpenCV는 이미지를 BGR 형식으로 읽기 때문에, matplotlib에서 올바른 색으로 보기 위해 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 컬러 이미지를 그레이스케일 이미지로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Sobel 필터를 사용하여 x축 방향 에지를 검출
# cv.CV_64F는 결과값을 64비트 실수형으로 계산하여 미세한 변화까지 보존하기 위함
# 1, 0 은 x방향 미분을 의미
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# Sobel 필터를 사용하여 y축 방향 에지를 검출
# 0, 1 은 y방향 미분을 의미
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# x축 에지와 y축 에지를 결합하여 전체 에지 강도를 계산
# magnitude는 각 픽셀에서 sqrt(sobel_x^2 + sobel_y^2)를 계산함
magnitude = cv.magnitude(sobel_x, sobel_y)

# magnitude 결과는 실수형이므로 화면에 표시하기 위해 절댓값 기반의 8비트 이미지로 변환
edge_uint8 = cv.convertScaleAbs(magnitude)

# 시각화를 위해 출력 창의 크기를 설정
plt.figure(figsize=(12, 6))

# 첫 번째 subplot에 원본 이미지를 표시
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 두 번째 subplot에 에지 강도 이미지를 흑백으로 표시
plt.subplot(1, 2, 2)
plt.imshow(edge_uint8, cmap='gray')
plt.title('Edge Magnitude Image')
plt.axis('off')

# subplot 간 간격을 자동 조정
plt.tight_layout()

# 결과 이미지를 화면에 출력
plt.show()