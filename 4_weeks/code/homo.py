import cv2 as cv                                  # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import numpy as np                               # 좌표 배열과 행렬 처리를 위해 numpy를 np라는 이름으로 불러옴
import matplotlib.pyplot as plt                  # 결과 시각화를 위해 matplotlib.pyplot을 plt로 불러옴

# 첫 번째 이미지를 컬러 형식으로 읽어서 변수에 저장
# 필요하면 img1.jpg 대신 img3.jpg 등으로 바꿔서 사용할 수 있음
img1 = cv.imread('img3.jpg')

# 두 번째 이미지를 컬러 형식으로 읽어서 img2 변수에 저장
# 실제 파일명이 imag2.jpg라면 이 부분을 imag2.jpg로 바꾸면 됨
img2 = cv.imread('img2.jpg')

# 두 이미지 중 하나라도 정상적으로 읽히지 않았는지 확인
if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")          # 오류 메시지를 출력
    exit()                                        # 프로그램 종료

# 특징점 검출을 위해 첫 번째 이미지를 그레이스케일로 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 특징점 검출을 위해 두 번째 이미지를 그레이스케일로 변환
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 객체를 생성
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 계산
kp1, des1 = sift.detectAndCompute(gray1, None)

# 두 번째 이미지에서 특징점과 기술자를 계산
kp2, des2 = sift.detectAndCompute(gray2, None)

# 기술자가 정상적으로 계산되었는지 확인
if des1 is None or des2 is None:
    print("특징점을 충분히 검출하지 못했습니다.")   # 오류 메시지를 출력
    exit()                                        # 프로그램 종료

# BFMatcher 객체를 생성
bf = cv.BFMatcher(cv.NORM_L2)

# knnMatch()를 사용하여 각 특징점마다 최근접 이웃 2개를 찾음
# k=2는 ratio test를 적용하기 위한 설정
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭만 저장할 빈 리스트를 생성
good_matches = []

# 모든 매칭 결과에 대해 반복
for pair in matches:
    # 매칭 쌍이 2개 모두 존재하는지 먼저 확인
    if len(pair) == 2:
        # 첫 번째 최근접 이웃을 m, 두 번째 최근접 이웃을 n에 저장
        m, n = pair

        # 거리 비율 테스트를 적용하여 신뢰할 수 있는 매칭만 선택
        # 첫 번째 후보가 두 번째 후보보다 충분히 더 가까우면 좋은 매칭으로 간주
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

# 호모그래피 계산에는 최소 4개의 좋은 매칭이 필요하므로 개수를 확인
if len(good_matches) < 4:
    print("좋은 매칭점이 부족하여 호모그래피를 계산할 수 없습니다.")
    exit()

# 첫 번째 이미지에서 좋은 매칭점의 좌표를 추출
# m.queryIdx는 첫 번째 이미지 특징점 인덱스임
# findHomography() 입력 형식에 맞게 (-1, 1, 2) 형태로 변환
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 두 번째 이미지에서 대응되는 좋은 매칭점의 좌표를 추출
# m.trainIdx는 두 번째 이미지 특징점 인덱스임
# findHomography() 입력 형식에 맞게 (-1, 1, 2) 형태로 변환
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC을 사용하여 호모그래피 행렬을 계산
# RANSAC은 이상치(outlier)의 영향을 줄이는 데 도움을 줌
# H는 계산된 호모그래피 행렬
# mask는 각 매칭점이 inlier인지 outlier인지 나타내는 마스크
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 호모그래피 계산이 정상적으로 되었는지 확인
if H is None:
    print("호모그래피 계산에 실패했습니다.")
    exit()

# 두 이미지의 높이와 너비를 구함
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 첫 번째 이미지를 두 번째 이미지 좌표계로 워핑
# 출력 크기는 문제 힌트에 따라 (w1 + w2, max(h1, h2))로 설정
warped = cv.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))

# 두 번째 이미지를 결과 영상의 왼쪽 상단에 배치
# 이렇게 하면 기준 이미지 위에 변환된 이미지가 정렬된 형태로 보일 수 있음
warped[0:h2, 0:w2] = img2

# RANSAC에서 살아남은 inlier 매칭만 시각화하기 위해 mask를 1차원 리스트로 변환
matches_mask = mask.ravel().tolist()

# drawMatches()를 사용하여 특징점 매칭 결과를 시각화
# matchesMask를 넣으면 RANSAC에서 inlier로 판단된 매칭만 그려짐
matching_result = cv.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 워핑된 결과 이미지는 BGR 형식이므로 matplotlib 출력을 위해 RGB로 변환
warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)

# 매칭 결과 이미지도 BGR 형식이므로 RGB로 변환
matching_result_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

# 전체 출력 창 크기를 설정
plt.figure(figsize=(18, 8))

# 1행 2열 중 첫 번째 위치에 매칭 결과를 출력
plt.subplot(1, 2, 1)

# 특징점 매칭 결과 이미지를 화면에 표시
plt.imshow(matching_result_rgb)

# 첫 번째 결과 제목을 설정
plt.title('Matching Result')

# 축 눈금을 보이지 않게 설정
plt.axis('off')

# 1행 2열 중 두 번째 위치에 워핑된 정렬 결과를 출력
plt.subplot(1, 2, 2)

# 호모그래피로 정렬된 이미지를 화면에 표시
plt.imshow(warped_rgb)

# 두 번째 결과 제목을 설정
plt.title('Warped Image')

# 축 눈금을 보이지 않게 설정
plt.axis('off')

# subplot 간격을 자동으로 정리
plt.tight_layout()

# 최종 결과를 화면에 출력
plt.show()