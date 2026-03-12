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
        gray,                                                  # 입력 그레이스케일 이미지
        CHECKERBOARD,                                          # 내부 코너 개수
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # 코너 검출에 성공한 경우만 처리
    if found:
        # 검출된 코너 좌표를 더 정밀하게 보정
        # sub-pixel 단위까지 코너 위치를 세밀하게 조정
        corners2 = cv2.cornerSubPix(
            gray,              # 그레이스케일 이미지
            corners,           # 초기 코너 좌표
            (11, 11),          # 탐색 윈도우 크기
            (-1, -1),          # zero zone
            criteria           # 종료 조건
        )

        # 현재 이미지에 대응되는 실제 3D 좌표를 저장
        # objp.copy()를 사용해 독립적인 배열로 저장
        objpoints.append(objp.copy())

        # 현재 이미지에서 검출된 정밀한 2D 코너 좌표를 저장
        imgpoints.append(corners2)

        # 검출된 코너를 시각화하기 위해 원본 이미지를 복사
        vis = img.copy()

        # 이미지 위에 검출된 체크보드 코너를 그림
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)

        # 파일 이름만 추출
        # 예: calibration_images/left01.jpg -> left01
        base = os.path.splitext(os.path.basename(fname))[0]

        # 코너가 표시된 이미지를 저장
        cv2.imwrite(f"result/{base}_corners.jpg", vis)

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

# 실제 3D 좌표(objpoints)와 이미지 2D 좌표(imgpoints)를 이용해
# 카메라 내부 행렬(K)과 왜곡 계수(dist)를 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,     # 실제 3D 좌표 목록
    imgpoints,     # 이미지 2D 코너 좌표 목록
    img_size,      # 이미지 크기 (width, height)
    None,          # 초기 camera matrix를 사용하지 않음
    None           # 초기 distortion coefficients를 사용하지 않음
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
    # alpha=1 이므로 가능한 한 원본 시야를 많이 유지
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,           # 기존 카메라 내부 행렬
        dist,        # 왜곡 계수
        (w, h),      # 원본 이미지 크기
        1,           # alpha=1: 최대한 전체 영역 유지
        (w, h)       # 출력 이미지 크기
    )

    # 계산된 K와 dist를 사용하여 이미지 왜곡을 보정
    undistorted = cv2.undistort(
        img,         # 원본 이미지
        K,           # 기존 카메라 내부 행렬
        dist,        # 왜곡 계수
        None,        # 새로운 왜곡 계수는 사용하지 않음
        new_K        # 새 카메라 행렬 사용
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