import cv2 as cv                              # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import matplotlib.pyplot as plt              # 이미지 출력용 matplotlib.pyplot을 plt로 불러옴
import numpy as np                           # 필요할 수 있는 배열 처리를 위해 numpy를 np로 불러옴

img = cv.imread('dabo.jpg')                  # dabo.jpg 이미지를 컬러로 읽어와 img에 저장

if img is None:                              # 이미지가 정상적으로 읽히지 않았는지 확인
    print("이미지를 불러올 수 없습니다.")       # 오류 메시지 출력
    exit()                                   # 프로그램 종료

original = img.copy()                        # 원본 이미지를 보존하기 위해 복사본을 만듦
line_img = img.copy()                        # 직선을 그릴 용도의 복사 이미지를 만듦

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # 컬러 이미지를 그레이스케일 이미지로 변환

edges = cv.Canny(gray, 100, 200)             # Canny 에지 검출을 사용하여 에지맵 생성

lines = cv.HoughLinesP(                      # 확률적 허프 변환을 사용하여 직선을 검출
    edges,                                   # 입력 에지 이미지
    1,                                       # rho 해상도: 1픽셀 단위
    np.pi / 180,                             # theta 해상도: 1도 단위(라디안)
    100,                                     # 직선으로 인정하기 위한 최소 투표 수
    minLineLength=50,                        # 검출할 직선의 최소 길이
    maxLineGap=10                            # 같은 직선으로 이어줄 최대 간격
)

if lines is not None:                        # 검출된 직선이 하나라도 있는지 확인
    for line in lines:                       # 검출된 각 직선에 대해 반복
        x1, y1, x2, y2 = line[0]             # 직선의 시작점과 끝점 좌표를 꺼냄
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 두께 2로 직선을 그림

original_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)   # matplotlib 출력용으로 BGR을 RGB로 변환
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)   # matplotlib 출력용으로 BGR을 RGB로 변환

plt.figure(figsize=(12, 6))                  # 전체 출력 창 크기를 설정

plt.subplot(1, 2, 1)                         # 1행 2열 중 첫 번째 영역 선택
plt.imshow(original_rgb)                     # 원본 이미지를 출력
plt.title('Original Image')                  # 첫 번째 이미지 제목 설정
plt.axis('off')                              # 축 눈금을 숨김

plt.subplot(1, 2, 2)                         # 1행 2열 중 두 번째 영역 선택
plt.imshow(line_img_rgb)                     # 직선이 그려진 이미지를 출력
plt.title('Detected Lines')                  # 두 번째 이미지 제목 설정
plt.axis('off')                              # 축 눈금을 숨김

plt.tight_layout()                           # 그래프 간격을 자동으로 정리
plt.show()                                   # 화면에 결과를 표시