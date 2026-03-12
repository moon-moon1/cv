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

    # 유효한 픽셀이 하나도 없으면 결과를 NaN으로 저장
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

# 유효한 결과가 부족하면 안내 문구 출력
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
    
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

cv2.imshow("Right with ROI", right_vis)

# disparity 컬러맵 이미지를 화면에 출력
cv2.imshow("Disparity Map", disparity_color)

# depth 컬러맵 이미지를 화면에 출력
cv2.imshow("Depth Map", depth_color)

# 키 입력이 들어올 때까지 창을 유지
cv2.waitKey(0)

# OpenCV 창을 모두 닫음
cv2.destroyAllWindows()