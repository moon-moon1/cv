# 컴퓨터 비전
본 저장소는 OpenCV를 이용한 컴퓨터비전 과제를 정리한 페이지입니다.  
과제는 총 3개로 구성되어 있습니다.

1. 체크보드 기반 카메라 캘리브레이션  
2. 이미지 Affine 변환  
3. Stereo Vision 기반 Disparity / Depth 추정  

각 과제마다 문제 설명, 핵심 코드, 중간 결과물, 최종 결과물, 전체 코드를 함께 정리하였습니다.

---

# 1번 과제: 체크보드 기반 카메라 캘리브레이션

## 1. 문제 설명
체크보드 패턴이 촬영된 여러 장의 이미지를 이용하여 카메라 캘리브레이션을 수행하였다.  
이미지에서 체크보드의 내부 코너를 검출한 뒤, 체크보드의 실제 3D 좌표와 이미지에서 검출된 2D 코너 좌표의 대응 관계를 이용하여 카메라 내부 행렬(Camera Matrix)과 왜곡 계수(Distortion Coefficients)를 계산하였다.  
이후 계산된 파라미터를 이용하여 왜곡 보정 결과를 시각화하였다.

### corner의 의미
corner는 체크보드에서 검정색 칸과 흰색 칸이 만나는 교차점이다.  
카메라 캘리브레이션에서는 이 코너들의 실제 위치와 이미지에서의 위치를 대응시켜 카메라 파라미터를 추정한다.

## 2. 요구사항
- 모든 이미지에서 체크보드 코너를 검출
- 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
- `cv2.calibrateCamera()`를 사용하여 카메라 내부 행렬 `K`와 왜곡 계수 `dist`를 계산
- `cv2.undistort()`를 사용하여 왜곡 보정 결과를 시각화

## 3. 사용한 주요 함수

### `cv2.findChessboardCorners()`
체크보드 이미지에서 내부 코너의 2D 이미지 좌표를 검출하는 함수이다.

### `cv2.calibrateCamera()`
실제 3D 좌표와 이미지 2D 좌표를 이용하여 카메라 내부 행렬과 왜곡 계수를 계산하는 함수이다.

### `cv2.undistort()`
계산된 카메라 파라미터를 이용해 왜곡된 이미지를 보정하는 함수이다.

## 4. 핵심 코드

### 체크보드 실제 좌표 생성
```python
CHECKERBOARD = (9, 6)
square_size = 25.0

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size
```

### 코너 검출
```python
found, corners = cv2.findChessboardCorners(
    gray,
    CHECKERBOARD,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)
```

### 카메라 캘리브레이션
```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)
```

### 왜곡 보정
```python
undistorted = cv2.undistort(img, K, dist, None, new_K)
```

## 5. 중간 결과물

### 체크보드 코너 검출 결과
체크보드 이미지에서 내부 코너를 검출한 뒤 `cv2.drawChessboardCorners()`를 사용하여 시각화하였다.

![corners1](result/left01_corners.jpg)
![corners2](result/left02_corners.jpg)
![corners3](result/left03_corners.jpg)

> 코너 검출에 실패한 이미지는 캘리브레이션에서 제외하였다.

## 6. 최종 결과물

### Camera Matrix

```text
여기에 Camera Matrix K 결과 붙여넣기
```

### Distortion Coefficients

```text
여기에 Distortion Coefficients 결과 붙여넣기
```

### 왜곡 보정 결과
원본 이미지와 왜곡 보정 이미지를 좌우로 연결하여 비교하였다.

![left01_compare](2_weeks/Chapter04_Calibrate/result/left01_compare.jpg)
![compare1](result/left01_compare.jpg)
![compare2](result/left02_compare.jpg)

### 해석
캘리브레이션을 통해 카메라 내부 행렬과 왜곡 계수를 추정할 수 있었고, 이를 사용하여 입력 이미지의 왜곡을 보정하였다.  
보정 결과, 원본 이미지에서 나타나던 렌즈 왜곡이 완화된 것을 확인할 수 있었다.

## 7. 전체 코드

```python
import cv2
import numpy as np
import glob
import os

# 체크보드의 "내부 코너" 개수를 지정
# (9, 6)은 가로 방향 내부 코너 9개, 세로 방향 내부 코너 6개를 의미
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기(mm)
square_size = 25.0

# 코너 위치를 더 정밀하게 보정할 때 사용할 종료 조건 설정
# 최대 30번 반복하거나, 정확도 변화가 0.001 이하가 되면 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체크보드의 실제 3D 좌표를 저장할 기본 배열 생성
# 내부 코너 개수만큼 행을 만들고, 각 점은 (x, y, z) 3차원 좌표를 가짐
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# z=0 평면 위에서 체크보드 코너들의 (x, y) 좌표를 생성
# 예: (0,0), (1,0), (2,0) ... 형태의 격자 좌표
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 실제 한 칸 크기가 25mm이므로 격자 좌표에 square_size를 곱해 실제 거리 단위로 변환
objp *= square_size

# 모든 이미지에서의 실제 3D 좌표들을 저장할 리스트
objpoints = []

# 모든 이미지에서 검출된 2D 이미지 코너 좌표들을 저장할 리스트
imgpoints = []

# calibration_images 폴더 안에서 left로 시작하는 jpg 파일들을 모두 불러옴
images = glob.glob("calibration_images/left*.jpg")

# 이미지 크기(width, height)를 저장할 변수
img_size = None

# 코너 검출 결과와 왜곡 보정 결과를 저장할 폴더 생성
os.makedirs("result", exist_ok=True)

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

# 불러온 모든 이미지 파일에 대해 반복
for fname in images:
    # 현재 이미지 파일을 읽음
    img = cv2.imread(fname)

    # 이미지를 읽지 못했다면 오류 메시지를 출력하고 다음 이미지로 넘어감
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {fname}")
        continue

    # 컬러 이미지를 그레이스케일 이미지로 변환
    # 체커보드 코너 검출은 보통 grayscale에서 수행
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 첫 번째 유효 이미지에서만 이미지 크기를 저장
    # gray.shape는 (height, width)이므로 [::-1]로 (width, height) 형태로 바꿈
    if img_size is None:
        img_size = gray.shape[::-1]

    # 그레이스케일 이미지에서 체크보드 내부 코너를 검출
    # found: 코너 검출 성공 여부(True/False)
    # corners: 검출된 코너 좌표들
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # 코너 검출에 성공한 경우만 처리
    if found:
        # 검출된 코너 좌표를 더 정밀하게 보정
        # sub-pixel 단위까지 코너 위치를 세밀하게 조정
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # 현재 이미지에 대응되는 실제 3D 좌표를 저장
        objpoints.append(objp.copy())

        # 현재 이미지에서 검출된 정밀한 2D 코너 좌표를 저장
        imgpoints.append(corners2)

        # 검출된 코너를 시각화하기 위해 원본 이미지를 복사
        vis = img.copy()

        # 이미지 위에 검출된 체크보드 코너를 그림
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)

        # 파일 이름만 추출
        base = os.path.splitext(os.path.basename(fname))[0]

        # 코너가 표시된 이미지를 저장
        cv2.imwrite(f"result/{base}_corners.jpg", vis)

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

# 실제 3D 좌표(objpoints)와 이미지 2D 좌표(imgpoints)를 이용해
# 카메라 내부 행렬(K)과 왜곡 계수(dist)를 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

# 계산된 카메라 내부 행렬 K를 출력
print("Camera Matrix K:")
print(K)

# 계산된 왜곡 계수 dist를 출력
print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------

# 다시 모든 이미지를 순회하면서 왜곡 보정 결과를 확인
for fname in images:
    # 원본 이미지를 읽음
    img = cv2.imread(fname)

    # 이미지를 읽지 못한 경우 다음 이미지로 넘어감
    if img is None:
        continue

    # 현재 이미지의 높이와 너비를 구함
    h, w = img.shape[:2]

    # 왜곡 보정 후 사용할 최적의 새 카메라 행렬을 계산
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (w, h),
        1,
        (w, h)
    )

    # 계산된 K와 dist를 사용하여 이미지 왜곡을 보정
    undistorted = cv2.undistort(
        img,
        K,
        dist,
        None,
        new_K
    )

    # 원본 이미지와 왜곡 보정 이미지를 좌우로 붙여 비교용 이미지 생성
    compare = np.hstack((img, undistorted))

    # 파일 이름만 추출
    base = os.path.splitext(os.path.basename(fname))[0]

    # 왜곡 보정된 이미지를 저장
    cv2.imwrite(f"result/{base}_undistorted.jpg", undistorted)

    # 원본/보정 비교 이미지를 저장
    cv2.imwrite(f"result/{base}_compare.jpg", compare)

    # 화면에 원본(왼쪽)과 보정 결과(오른쪽)를 표시
    cv2.imshow("Original (Left) | Undistorted (Right)", compare)

    # 키 입력을 기다림
    key = cv2.waitKey(0)

    # ESC 키(ASCII 27)를 누르면 반복을 중단하고 종료
    if key == 27:
        break

# 모든 OpenCV 창을 닫음
cv2.destroyAllWindows()
```

---

# 2번 과제: 이미지 Affine 변환

## 1. 문제 설명
한 장의 이미지에 대해 중심 기준 회전, 크기 조절, 평행이동을 적용하는 과제이다.  
이미지 중심을 기준으로 +30도 회전시키고, 동시에 크기를 0.8배로 조절한 뒤, 결과를 x축 방향으로 +80px, y축 방향으로 -40px 만큼 평행이동하였다.

## 2. 요구사항
- 이미지 중심 기준으로 +30도 회전
- 회전과 동시에 크기를 0.8배로 조절
- 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px 평행이동

## 3. 사용한 주요 함수

### `cv2.getRotationMatrix2D()`
이미지 중심 기준 회전 + 크기 조절 행렬을 생성하는 함수이다.

### `cv2.warpAffine()`
Affine 변환 행렬을 실제 이미지에 적용하는 함수이다.

## 4. 핵심 코드

### 회전 + 크기 조절 행렬 생성
```python
center = (w / 2, h / 2)
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)
```

### 평행이동 반영
```python
tx = 80
ty = -40
M[0, 2] += tx
M[1, 2] += ty
```

### Affine 변환 적용
```python
result = cv2.warpAffine(img, M, (w, h))
```

## 5. 중간 결과물
회전 및 크기 조절 행렬 `M`을 생성한 뒤, 마지막 열 값을 수정하여 평행이동을 반영하였다.

## 6. 최종 결과물

### 변환 결과 이미지
![affine_result](transformed.jpg)

### 해석
`cv2.getRotationMatrix2D()`를 사용하여 이미지 중심 기준 회전과 크기 조절을 동시에 수행하였다.  
이후 행렬의 마지막 열 값을 수정하여 x축과 y축 방향 평행이동을 추가하였다.  
최종적으로 `cv2.warpAffine()`를 적용하여 변환된 이미지를 얻을 수 있었다.

## 7. 전체 코드

```python
import cv2
import numpy as np
import sys

# 이미지 파일 "rose.png"를 읽어서 img 변수에 저장
img = cv2.imread("rose.png")

# 이미지를 제대로 읽지 못했으면 오류 메시지를 출력하고 프로그램 종료
if img is None:
    sys.exit("이미지를 불러올 수 없습니다.")

# 이미지의 높이(h)와 너비(w)를 구함
# img.shape[:2]는 (height, width)를 반환
h, w = img.shape[:2]

# 이미지의 중심 좌표를 계산
# 회전은 보통 중심점을 기준으로 수행하므로 중심 좌표가 필요함
center = (w / 2, h / 2)

# -----------------------------
# 1. 회전 + 크기 조절 행렬 생성
# -----------------------------

# 회전 각도를 설정
angle = 30

# 크기 조절 비율을 설정
scale = 0.8

# 이미지 중심(center)을 기준으로
M = cv2.getRotationMatrix2D(center, angle, scale)

# -----------------------------
# 2. 평행이동 반영
# -----------------------------

# x축 방향 평행이동 값을 설정
tx = 80

# y축 방향 평행이동 값을 설정
ty = -40

# 변환 행렬 M의 [0, 2] 원소는 x축 이동 성분이므로 여기에 tx를 더함
M[0, 2] += tx

# 변환 행렬 M의 [1, 2] 원소는 y축 이동 성분이므로 여기에 ty를 더함
M[1, 2] += ty

# -----------------------------
# 3. Affine 변환 적용
# -----------------------------

# cv2.warpAffine()를 사용하여 원본 이미지에 변환 행렬 M을 적용
# 출력 이미지 크기는 원본과 같은 (w, h)로 설정
result = cv2.warpAffine(img, M, (w, h))

# 원본 이미지(img)와 변환된 이미지(result)를 좌우로 붙여서 하나의 이미지로 만듦
img2 = np.hstack((img, result))

# 원본과 변환 이미지를 나란히 붙인 비교용 이미지를 화면에 출력
cv2.imshow("Result", img2)

# 변환된 결과 이미지만 "transformed.jpg"라는 이름으로 저장
cv2.imwrite("transformed.jpg", result)

# 키보드 입력이 들어올 때까지 창을 유지
cv2.waitKey(0)

# OpenCV로 연 모든 창을 닫음
cv2.destroyAllWindows()
```

---

# 3번 과제: Stereo Vision 기반 Disparity / Depth 추정

## 1. 문제 설명
같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용하여 깊이 정보를 추정하였다.  
두 이미지에서 같은 물체가 좌우로 얼마나 이동해 보이는지(disparity)를 계산하고, 이를 이용해 물체와 카메라 사이의 거리(depth)를 계산하였다.

## 2. 요구사항
- 입력 이미지를 그레이스케일로 변환한 뒤 `cv2.StereoBM_create()`를 사용하여 disparity map 계산
- `Disparity > 0` 인 픽셀만 사용하여 depth map 계산
- ROI `Painting`, `Frog`, `Teddy` 각각에 대해 평균 disparity와 평균 depth를 계산
- 세 ROI 중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해석

## 3. 개념 정리

### Disparity
왼쪽 이미지와 오른쪽 이미지에서 같은 물체가 나타나는 픽셀 위치 차이이다.  
일반적으로 disparity 값이 클수록 카메라에 더 가까운 물체이다.

### Depth
disparity를 이용해 계산한 실제 거리 정보이다.  
본 과제에서는 다음 공식을 사용하였다.

\[
Z = \frac{fB}{d}
\]

- `f`: focal length
- `B`: baseline
- `d`: disparity

depth 값이 클수록 더 먼 물체로 해석할 수 있다.

## 4. 사용한 주요 함수

### `cv2.StereoBM_create()`
스테레오 매칭을 통해 disparity map을 계산하는 함수이다.

### `cv2.applyColorMap()`
disparity map과 depth map을 컬러로 시각화하는 데 사용하였다.

## 5. 핵심 코드

### 그레이스케일 변환
```python
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
```

### disparity 계산
```python
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
```

### depth 계산
```python
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity, dtype=np.float32)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]
```

### ROI별 평균 disparity / depth 계산
```python
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
```

## 6. 중간 결과물

### ROI 설정 결과
왼쪽 이미지와 오른쪽 이미지에 관심 영역(ROI)을 표시하였다.

![left_roi](outputs/left_with_roi.png)
![right_roi](outputs/right_with_roi.png)

### disparity 시각화 결과
disparity map은 정규화 후 컬러맵을 적용하여 시각화하였다.

![disparity](outputs/disparity_color.png)

## 7. 최종 결과물

### depth 시각화 결과
![depth](outputs/depth_color.png)

### ROI별 평균 disparity / depth
아래에 실행 결과를 붙여넣으면 된다.

```text
여기에 ROI별 평균 disparity / depth 결과 붙여넣기

예시:
=== ROI별 평균 Disparity / Depth ===
Painting
  Mean Disparity : ...
  Mean Depth     : ... m

Frog
  Mean Disparity : ...
  Mean Depth     : ... m

Teddy
  Mean Disparity : ...
  Mean Depth     : ... m
```

### 가까운 영역 / 먼 영역 해석
아래에 실행 결과를 붙여넣으면 된다.

```text
여기에 가까운 영역 / 먼 영역 결과 붙여넣기

예시:
=== 해석 ===
가장 가까운 영역: ...
가장 먼 영역: ...
```

### 해석
disparity가 클수록 물체는 카메라에 더 가까우며, depth가 클수록 더 멀다.  
ROI별 평균 disparity와 평균 depth를 비교한 결과, 가장 큰 평균 disparity를 가지는 영역이 가장 가까운 영역이고, 가장 큰 평균 depth를 가지는 영역이 가장 먼 영역으로 해석된다.

## 8. 전체 코드

```python
import cv2
import numpy as np
from pathlib import Path

# 출력 결과를 저장할 폴더 경로를 설정
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 왼쪽 이미지를 컬러로 읽어서 left_color 변수에 저장
left_color = cv2.imread("left.png")

# 오른쪽 이미지를 컬러로 읽어서 right_color 변수에 저장
right_color = cv2.imread("right.png")

# 둘 중 하나라도 이미지를 읽지 못하면 예외를 발생시켜 프로그램 종료
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# focal length 값을 설정
f = 700.0

# 두 카메라 사이 거리를 설정
B = 0.12

# ROI(관심 영역)를 딕셔너리 형태로 정의
# 형식은 "이름": (x, y, w, h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# -----------------------------
# 0. 그레이스케일 변환
# -----------------------------

# 왼쪽 컬러 이미지를 그레이스케일 이미지로 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)

# 오른쪽 컬러 이미지를 그레이스케일 이미지로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------

# StereoBM 객체를 생성
# numDisparities는 16의 배수여야 하며, blockSize는 홀수여야 함
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)

# 왼쪽/오른쪽 그레이스케일 이미지로 disparity를 계산
# 결과는 16배 스케일된 정수값이므로 float32로 바꾸고 16으로 나눔
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------

# disparity가 0보다 큰 픽셀만 유효하다고 판단
valid_mask = disparity > 0

# disparity와 같은 크기의 depth 맵을 0으로 초기화
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한 disparity 픽셀에 대해서만 깊이 값을 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------

# 각 ROI의 결과를 저장할 딕셔너리를 생성
results = {}

# 정의한 모든 ROI에 대해 반복
for name, (x, y, w, h) in rois.items():
    # disparity 맵에서 현재 ROI 영역만 잘라냄
    roi_disp = disparity[y:y+h, x:x+w]

    # depth 맵에서 현재 ROI 영역만 잘라냄
    roi_depth = depth_map[y:y+h, x:x+w]

    # 현재 ROI 안에서 disparity가 0보다 큰 픽셀만 유효하다고 판단
    roi_valid = roi_disp > 0

    # 유효한 픽셀이 하나라도 있으면 평균 disparity와 평균 depth를 계산
    if np.any(roi_valid):
        mean_disp = float(np.mean(roi_disp[roi_valid]))
        mean_depth = float(np.mean(roi_depth[roi_valid]))
    else:
        mean_disp = float("nan")
        mean_depth = float("nan")

    # 현재 ROI의 결과를 딕셔너리에 저장
    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------

# 제목을 출력
print("=== ROI별 평균 Disparity / Depth ===")

# 각 ROI의 평균 disparity와 평균 depth를 출력
for name, values in results.items():
    print(f"{name}")
    print(f"  Mean Disparity : {values['mean_disparity']:.4f}")
    print(f"  Mean Depth     : {values['mean_depth']:.4f} m")

# 평균 disparity가 NaN이 아닌 ROI만 따로 모음
valid_results_disp = {
    name: v for name, v in results.items()
    if not np.isnan(v["mean_disparity"])
}

# 평균 depth가 NaN이 아닌 ROI만 따로 모음
valid_results_depth = {
    name: v for name, v in results.items()
    if not np.isnan(v["mean_depth"])
}

# 유효한 결과가 존재하면 가장 가까운 영역과 가장 먼 영역을 계산
if valid_results_disp and valid_results_depth:
    # disparity가 클수록 더 가까우므로 평균 disparity가 가장 큰 ROI를 선택
    nearest_obj = max(valid_results_disp.items(), key=lambda x: x[1]["mean_disparity"])[0]

    # depth가 클수록 더 멀므로 평균 depth가 가장 큰 ROI를 선택
    farthest_obj = max(valid_results_depth.items(), key=lambda x: x[1]["mean_depth"])[0]

    # 해석 결과를 출력
    print("\n=== 해석 ===")
    print(f"가장 가까운 영역: {nearest_obj}")
    print(f"가장 먼 영역: {farthest_obj}")
else:
    print("\n유효한 ROI 결과가 부족하여 가까운/먼 영역을 판별할 수 없습니다.")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# disparity를 복사해서 시각화용 임시 배열 생성
disp_tmp = disparity.copy()

# disparity가 0 이하인 값은 유효하지 않으므로 NaN으로 바꿈
disp_tmp[disp_tmp <= 0] = np.nan

# 모든 disparity 값이 NaN이면 유효한 값이 없는 것이므로 예외 처리
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 너무 작은 값과 너무 큰 값의 영향을 줄이기 위해 최소값으로 사용
d_min = np.nanpercentile(disp_tmp, 5)

# 95퍼센타일 값을 최대값으로 사용
d_max = np.nanpercentile(disp_tmp, 95)

# 최대값이 최소값보다 작거나 같으면 나눗셈 오류 방지를 위해 아주 작은 값을 더함
if d_max <= d_min:
    d_max = d_min + 1e-6

# disparity 값을 0~1 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)

# 정규화 결과가 범위를 넘지 않도록 0~1로 자름
disp_scaled = np.clip(disp_scaled, 0, 1)

# 8비트 시각화 영상을 만들기 위해 uint8 배열을 생성
disp_vis = np.zeros_like(disparity, dtype=np.uint8)

# NaN이 아닌 유효 disparity 위치를 마스크로 생성
valid_disp = ~np.isnan(disp_tmp)

# 유효한 위치에 대해 0~255 범위의 밝기값으로 변환
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 컬러맵을 적용하여 disparity를 색상 이미지로 변환
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# depth 시각화용 8비트 배열을 생성
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

# 유효한 depth 값이 하나라도 있으면 시각화를 진행
if np.any(valid_mask):
    # 유효한 depth 값들만 따로 추출
    depth_valid = depth_map[valid_mask]

    # 5퍼센타일 값을 최소값으로 사용
    z_min = np.percentile(depth_valid, 5)

    # 95퍼센타일 값을 최대값으로 사용
    z_max = np.percentile(depth_valid, 95)

    # 최대값이 최소값보다 작거나 같으면 나눗셈 오류 방지를 위해 아주 작은 값을 더함
    if z_max <= z_min:
        z_max = z_min + 1e-6

    # depth 값을 0~1 범위로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)

    # 정규화 결과를 0~1 범위로 제한
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 값이 클수록 더 멀기 때문에
    # 가까울수록 빨강이 되게 하려면 반전시켜야 함
    depth_scaled = 1.0 - depth_scaled

    # 유효한 위치에 대해 0~255 범위의 밝기값으로 변환
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# 컬러맵을 적용하여 depth를 색상 이미지로 변환
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left 이미지에 ROI 표시
# -----------------------------

# 왼쪽 원본 이미지를 복사해서 ROI 표시용 이미지 생성
left_vis = left_color.copy()
right_vis = right_color.copy()

# 모든 ROI에 대해 반복
for name, (x, y, w, h) in rois.items():
    # ROI 사각형을 초록색으로 그림
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI 이름을 사각형 위쪽에 초록색 글씨로 표시
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 오른쪽 이미지에도 ROI 사각형을 그림
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 오른쪽 이미지에도 ROI 이름을 표시
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------

# 왼쪽 이미지에 ROI를 표시한 결과를 저장
cv2.imwrite(str(output_dir / "left_with_roi.png"), left_vis)

# 오른쪽 이미지에 ROI를 표시한 결과를 저장
cv2.imwrite(str(output_dir / "right_with_roi.png"), right_vis)

# disparity 컬러 시각화 결과를 저장
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)

# depth 컬러 시각화 결과를 저장
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)

# -----------------------------
# 9. 출력
# -----------------------------

# ROI가 표시된 왼쪽 이미지를 화면에 출력
cv2.imshow("Left with ROI", left_vis)

# ROI가 표시된 오른쪽 이미지를 화면에 출력
cv2.imshow("Right with ROI", right_vis)

# disparity 컬러맵 이미지를 화면에 출력
cv2.imshow("Disparity Map", disparity_color)

# depth 컬러맵 이미지를 화면에 출력
cv2.imshow("Depth Map", depth_color)

# 키 입력이 들어올 때까지 창을 유지
cv2.waitKey(0)

# OpenCV 창을 모두 닫음
cv2.destroyAllWindows()
```

---

# 정리

1. 체크보드 코너 검출을 이용한 카메라 캘리브레이션  
2. 회전, 크기 조절, 평행이동을 포함한 Affine 변환  
3. 스테레오 영상의 disparity 계산 및 depth 추정  
