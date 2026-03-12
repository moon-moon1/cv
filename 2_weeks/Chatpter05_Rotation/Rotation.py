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