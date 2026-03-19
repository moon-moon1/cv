# 컴퓨터 비전
본 저장소는 OpenCV를 이용한 컴퓨터비전 과제를 정리한 페이지입니다.  
과제는 총 3개로 구성되어 있습니다.

1. Sobel 필터 기반 에지 강도 시각화  
2. Canny + Hough 변환 기반 직선 검출  
3. GrabCut 기반 객체 추출  

각 과제마다 문제 설명, 핵심 코드, 결과물, 전체 코드를 함께 정리하였습니다.

---

# 1번 과제: Sobel 필터 기반 에지 강도 시각화

## 1. 문제 설명
`edgeDetectionImage` 이미지를 그레이스케일로 변환한 뒤, Sobel 필터를 사용하여 x축 방향과 y축 방향의 에지를 검출하였다.  
이후 두 방향의 에지 정보를 결합하여 에지 강도(edge magnitude) 이미지를 생성하고, 원본 이미지와 함께 시각화하였다.

## 2. 요구사항
- `cv.imread()`를 사용하여 이미지를 불러옴
- `cv.cvtColor()`를 사용하여 그레이스케일로 변환
- `cv.Sobel()`을 사용하여 x축과 y축 방향의 에지를 검출
- `cv.magnitude()`를 사용하여 에지 강도를 계산
- `cv.convertScaleAbs()`를 사용하여 에지 강도 이미지를 `uint8`로 변환
- Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화

## 3. 사용한 주요 함수

### `cv.Sobel()`
입력 이미지의 밝기 변화를 미분하여 특정 방향의 에지를 검출하는 함수이다.  
x방향과 y방향 각각에 대해 적용하여 수평 및 수직 방향의 경계 정보를 얻을 수 있다.

### `cv.magnitude()`
x축 방향 에지와 y축 방향 에지를 결합하여 최종 에지 강도를 계산하는 함수이다.  
각 픽셀에서 `sqrt(x^2 + y^2)` 형태로 계산된다.

### `cv.convertScaleAbs()`
실수형으로 계산된 에지 강도 이미지를 절댓값 기반의 8비트 이미지로 변환하는 함수이다.

## 4. 핵심 코드

### 그레이스케일 변환
~~~python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
~~~

### x축 / y축 Sobel 에지 검출
~~~python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
~~~

### 에지 강도 계산
~~~python
magnitude = cv.magnitude(sobel_x, sobel_y)
edge_uint8 = cv.convertScaleAbs(magnitude)
~~~

## 5. 결과물

### 에지 강도 시각화 결과
원본 이미지와 Sobel 기반 에지 강도 이미지를 좌우로 배치하여 비교하였다.

<p align="center">
  <img src="https://raw.githubusercontent.com/moon-moon1/cv/main/3_weeks/Result/Sobel.png" width="80%">
</p>


### 해석
Sobel 필터를 이용하여 x축과 y축 방향의 밝기 변화량을 각각 구한 뒤, 이를 결합하여 전체 에지 강도를 계산하였다.  
그 결과 물체의 경계나 밝기 변화가 큰 부분이 더 강한 값으로 나타나는 에지 강도 이미지를 얻을 수 있었다.

## 6. 전체 코드

~~~python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# Sobel 필터를 사용하여 y축 방향 에지를 검출
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# x축 에지와 y축 에지를 결합하여 전체 에지 강도를 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 화면에 표시하기 위해 8비트 이미지로 변환
edge_uint8 = cv.convertScaleAbs(magnitude)

# 출력 창 크기 설정
plt.figure(figsize=(12, 6))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 에지 강도 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(edge_uint8, cmap='gray')
plt.title('Edge Magnitude Image')
plt.axis('off')

# 그래프 간격 자동 조정
plt.tight_layout()

# 결과 출력
plt.show()
~~~

---

# 2번 과제: Canny + Hough 변환 기반 직선 검출

## 1. 문제 설명
`dabo` 이미지에 대해 Canny 에지 검출을 사용하여 에지맵을 생성한 뒤, Hough 변환을 이용하여 직선을 검출하였다.  
검출된 직선은 원본 이미지 위에 빨간색으로 표시하였으며, 원본 이미지와 직선이 그려진 결과 이미지를 함께 시각화하였다.

## 2. 요구사항
- `cv.Canny()`를 사용하여 에지맵 생성
- `cv.HoughLinesP()`를 사용하여 직선 검출
- `cv.line()`을 사용하여 검출된 직선을 원본 이미지에 그림
- Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화

## 3. 사용한 주요 함수

### `cv.Canny()`
이미지에서 경계를 검출하여 에지맵을 생성하는 함수이다.  
강한 밝기 변화가 있는 부분을 선 형태로 추출할 수 있다.

### `cv.HoughLinesP()`
확률적 허프 변환을 사용하여 에지맵에서 직선을 검출하는 함수이다.  
직선의 시작점과 끝점을 반환하므로 바로 그리기가 가능하다.

### `cv.line()`
이미지 위에 선분을 그리는 함수이다.  
검출된 직선을 시각화하는 데 사용하였다.

## 4. 핵심 코드

### Canny 에지맵 생성
~~~python
edges = cv.Canny(gray, 100, 200)
~~~

### Hough 직선 검출
~~~python
lines = cv.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    100,
    minLineLength=50,
    maxLineGap=10
)
~~~

### 검출된 직선 그리기
~~~python
cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
~~~

## 5. 결과물

### 직선 검출 결과
원본 이미지와 검출된 직선을 빨간색으로 표시한 이미지를 좌우로 비교하였다.

<p align="center">
  <img src="https://raw.githubusercontent.com/moon-moon1/cv/main/3_weeks/Result/Canny.png" width="80%">
</p>

### 해석
Canny 에지 검출로 경계선을 추출한 뒤, HoughLinesP를 이용해 직선 형태의 에지를 선분으로 검출하였다.  
그 결과 이미지 내부의 주요 직선 구조를 시각적으로 확인할 수 있었다.

## 6. 전체 코드

~~~python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# dabo 이미지를 컬러로 읽어옴
img = cv.imread('dabo.jpg')

# 이미지가 정상적으로 불러와졌는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 원본 이미지와 직선 그리기용 이미지를 각각 복사
original = img.copy()
line_img = img.copy()

# 컬러 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 에지 검출을 사용하여 에지맵 생성
edges = cv.Canny(gray, 100, 200)

# 확률적 허프 변환을 사용하여 직선을 검출
lines = cv.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    100,
    minLineLength=50,
    maxLineGap=10
)

# 검출된 직선이 있으면 반복하면서 그림
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# matplotlib 출력용으로 BGR 이미지를 RGB로 변환
original_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

# 출력 창 크기 설정
plt.figure(figsize=(12, 6))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('off')

# 직선 검출 결과 출력
plt.subplot(1, 2, 2)
plt.imshow(line_img_rgb)
plt.title('Detected Lines')
plt.axis('off')

# 그래프 간격 자동 조정
plt.tight_layout()

# 결과 출력
plt.show()
~~~

---

# 3번 과제: GrabCut 기반 객체 추출

## 1. 문제 설명
`coffee cup` 이미지에서 사용자가 지정한 사각형 영역을 기준으로 GrabCut 알고리즘을 적용하여 객체를 추출하였다.  
GrabCut 수행 후 생성된 마스크를 사용하여 배경을 제거하고, 원본 이미지에서 객체만 남긴 결과를 시각화하였다.

## 2. 요구사항
- `cv.grabCut()`를 사용하여 대화식 분할 수행
- 초기 사각형 영역은 `(x, y, width, height)` 형식으로 설정
- 마스크를 사용하여 원본 이미지에서 배경 제거
- Matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지를 나란히 시각화

## 3. 개념 정리

### GrabCut 마스크 값의 의미
GrabCut의 마스크는 픽셀마다 다음 4가지 상태 중 하나를 가진다.

- `cv.GC_BGD` : 확실한 배경
- `cv.GC_FGD` : 확실한 전경
- `cv.GC_PR_BGD` : 배경일 가능성이 높은 영역
- `cv.GC_PR_FGD` : 전경일 가능성이 높은 영역

최종적으로는 이 4가지 값을 배경(0)과 전경(1)으로 다시 변환하여 객체만 추출한다.

### 초기 사각형의 의미
GrabCut은 사용자가 지정한 사각형을 기반으로 객체가 대략 어디에 있는지 추정한다.  
사각형 바깥은 배경으로 간주하고, 사각형 안쪽은 객체가 존재할 가능성이 있는 영역으로 보아 분할을 시작한다.

## 4. 사용한 주요 함수

### `cv.grabCut()`
초기 사각형 또는 초기 마스크를 바탕으로 배경과 전경을 분리하는 함수이다.

### `np.where()`
GrabCut 결과 마스크를 0과 1 형태의 이진 마스크로 변환하는 데 사용하였다.

## 5. 핵심 코드

### 초기 마스크와 모델 생성
~~~python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
~~~

### 초기 사각형 설정
~~~python
rect = (50, 30, 250, 300)
~~~

### GrabCut 수행
~~~python
cv.grabCut(
    img,
    mask,
    rect,
    bgdModel,
    fgdModel,
    5,
    cv.GC_INIT_WITH_RECT
)
~~~

### 최종 마스크 생성
~~~python
mask2 = np.where(
    (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
    1,
    0
).astype('uint8')
~~~

## 6. 결과물

### 객체 추출 결과
원본 이미지, GrabCut 마스크 이미지, 배경이 제거된 객체 추출 결과를 나란히 비교하였다.

<p align="center">
  <img src="https://raw.githubusercontent.com/moon-moon1/cv/main/3_weeks/Result/Grabcut.png" width="80%">
</p>

### 해석
GrabCut 알고리즘을 사용하여 사각형 내부를 중심으로 객체와 배경을 분리하였다.  
이후 마스크 값을 전경과 배경으로 다시 변환하고, 이를 원본 이미지에 적용하여 배경을 제거한 객체 추출 결과를 얻을 수 있었다.

## 7. 전체 코드

~~~python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# coffee_cup 이미지를 읽어서 img 변수에 저장
img = cv.imread('coffee_cup.jpg')

# 이미지가 정상적으로 불러와졌는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# matplotlib에서 올바른 색으로 보기 위해 BGR 이미지를 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# GrabCut 결과를 저장할 초기 마스크를 생성
mask = np.zeros(img.shape[:2], np.uint8)

# GrabCut이 사용할 배경 모델과 전경 모델을 0으로 초기화
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 객체가 포함되도록 초기 사각형 영역을 설정
rect = (50, 30, 250, 300)

# 사각형을 기반으로 GrabCut 알고리즘을 수행
cv.grabCut(
    img,
    mask,
    rect,
    bgdModel,
    fgdModel,
    5,
    cv.GC_INIT_WITH_RECT
)

# GrabCut의 4가지 마스크 값을 전경(1)과 배경(0)으로 변환
mask2 = np.where(
    (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
    1,
    0
).astype('uint8')

# 마스크를 이용해 배경을 제거하고 객체만 남김
result = img_rgb * mask2[:, :, np.newaxis]

# 마스크 시각화를 위해 0/1 값을 0/255 값으로 변환
mask_display = mask2 * 255

# 출력 창 크기 설정
plt.figure(figsize=(15, 5))

# 원본 이미지 출력
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 마스크 이미지 출력
plt.subplot(1, 3, 2)
plt.imshow(mask_display, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

# 배경 제거 결과 이미지 출력
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title('Background Removed')
plt.axis('off')

# 그래프 간격 자동 조정
plt.tight_layout()

# 결과 출력
plt.show()
~~~

---

# 정리

1. Sobel 필터를 이용한 에지 강도 시각화  
2. Canny + Hough 변환을 이용한 직선 검출  
3. GrabCut을 이용한 객체 추출 및 배경 제거
