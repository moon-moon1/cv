import cv2 as cv                                      # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import numpy as np                                   # 배열 및 수치 연산을 위해 numpy를 np라는 이름으로 불러옴
import matplotlib.pyplot as plt                      # 이미지 시각화를 위해 matplotlib.pyplot을 plt로 불러옴

img = cv.imread('coffee_cup.jpg')                    # coffee_cup.jpg 이미지를 읽어서 img 변수에 저장

if img is None:                                      # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지를 불러올 수 없습니다.")              # 이미지 로드 실패 메시지 출력
    exit()                                           # 프로그램 종료

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)         # matplotlib에서 올바른 색으로 보기 위해 BGR 이미지를 RGB로 변환

mask = np.zeros(img.shape[:2], np.uint8)             # 이미지와 같은 높이, 너비를 가지는 초기 마스크를 0으로 생성

bgdModel = np.zeros((1, 65), np.float64)             # GrabCut이 사용할 배경 모델 배열을 0으로 초기화
fgdModel = np.zeros((1, 65), np.float64)             # GrabCut이 사용할 전경 모델 배열을 0으로 초기화

rect = (100, 90, 1100, 800)                            # 초기 사각형 영역을 (x, y, width, height) 형식으로 설정 ,x,y : 사각형 시작 좌표 , width,height : 사각형(객체) 크기

cv.grabCut(                                          # GrabCut 알고리즘을 실행하여 객체와 배경을 분리
    img,                                             # 입력 원본 이미지
    mask,                                            # 분할 결과가 저장될 마스크
    rect,                                            # 사용자가 지정한 초기 사각형 영역
    bgdModel,                                        # 배경 모델
    fgdModel,                                        # 전경 모델
    5,                                               # 반복 횟수 설정
    cv.GC_INIT_WITH_RECT                             # 사각형 영역을 초기값으로 사용하겠다는 옵션
)

mask2 = np.where(                                    # GrabCut 결과 마스크를 0과 1로 변환
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),    # 확실한 배경 또는 배경일 가능성이 높은 영역이면
    0,                                               # 0으로 설정하여 제거 대상이 되게 함
    1                                                # 나머지는 1로 설정하여 객체로 남김
).astype('uint8')                                    # 결과를 uint8 형식으로 변환

result = img_rgb * mask2[:, :, np.newaxis]           # 2차원 마스크를 3채널로 확장하여 원본 RGB 이미지에 곱해 배경 제거

mask_display = mask2 * 255                           # 마스크를 시각화하기 위해 0/1 값을 0/255 값으로 변환

plt.figure(figsize=(15, 5))                          # 전체 출력 창 크기를 설정

plt.subplot(1, 3, 1)                                 # 1행 3열 중 첫 번째 영역 선택
plt.imshow(img_rgb)                                  # 원본 이미지를 출력
plt.title('Original Image')                          # 첫 번째 이미지 제목 설정
plt.axis('off')                                      # 축을 숨김

plt.subplot(1, 3, 2)                                 # 1행 3열 중 두 번째 영역 선택
plt.imshow(mask_display, cmap='gray')                # 마스크 이미지를 흑백으로 출력
plt.title('Mask Image')                              # 두 번째 이미지 제목 설정
plt.axis('off')                                      # 축을 숨김

plt.subplot(1, 3, 3)                                 # 1행 3열 중 세 번째 영역 선택
plt.imshow(result)                                   # 배경이 제거된 결과 이미지를 출력
plt.title('Background Removed')                      # 세 번째 이미지 제목 설정
plt.axis('off')                                      # 축을 숨김

plt.tight_layout()                                   # 서브플롯 간격을 자동으로 정리
plt.show()                                           # 결과를 화면에 출력