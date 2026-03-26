import cv2 as cv                                  # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import matplotlib.pyplot as plt                  # 결과 시각화를 위해 matplotlib.pyplot을 plt로 불러옴

# 첫 번째 이미지를 컬러 형식으로 읽어서 img1 변수에 저장
img1 = cv.imread('mot_color70.jpg')

# 두 번째 이미지를 컬러 형식으로 읽어서 img2 변수에 저장
img2 = cv.imread('mot_color83.jpg')

# 두 이미지 중 하나라도 정상적으로 읽히지 않았는지 확인
if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")          # 오류 메시지를 출력
    exit()                                        # 프로그램 종료

# 특징점 검출은 보통 그레이스케일 이미지에서 수행하므로
# 첫 번째 이미지를 그레이스케일로 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 두 번째 이미지를 그레이스케일로 변환
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 객체를 생성
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 계산
# kp1은 특징점 목록, des1은 특징점 기술자 배열
kp1, des1 = sift.detectAndCompute(gray1, None)

# 두 번째 이미지에서 특징점과 기술자를 계산
# kp2는 특징점 목록, des2는 특징점 기술자 배열
kp2, des2 = sift.detectAndCompute(gray2, None)

# 기술자가 정상적으로 계산되었는지 확인
# 이미지에 특징점이 거의 없으면 기술자가 None이 될 수 있음
if des1 is None or des2 is None:
    print("특징점을 충분히 검출하지 못했습니다.")   # 오류 메시지를 출력
    exit()                                        # 프로그램 종료

# FLANN 매칭을 위한 인덱스 파라미터를 설정
# algorithm=1은 KD-Tree 방식을 의미하며 SIFT처럼 float descriptor에 적합함
# trees=5는 KD-Tree 개수를 의미하며 보통 4~8 정도를 많이 사용함
index_params = dict(algorithm=1, trees=5)

# FLANN 탐색 파라미터를 설정
# checks 값이 클수록 더 많이 탐색하여 정확도가 올라갈 수 있지만 속도는 느려질 수 있음
search_params = dict(checks=50)

# FLANN 기반 매처 객체를 생성
flann = cv.FlannBasedMatcher(index_params, search_params)

# knnMatch()를 사용하여 각 특징점에 대해 가장 가까운 이웃 2개를 찾음
# k=2로 설정한 이유는 ratio test를 적용하기 위해서임
matches = flann.knnMatch(des1, des2, k=2)

# 좋은 매칭만 저장할 빈 리스트를 생성
good_matches = []

# 모든 매칭 쌍에 대해 반복
for pair in matches:
    # 간혹 매칭 결과가 2개 미만인 경우가 있을 수 있으므로 길이를 먼저 확인
    if len(pair) == 2:
        # 가장 가까운 매칭을 m, 두 번째로 가까운 매칭을 n에 저장
        m, n = pair

        # Lowe의 ratio test를 적용
        # 첫 번째 후보의 거리가 두 번째 후보 거리의 0.75배보다 작으면
        # 더 신뢰할 수 있는 매칭이라고 판단하여 good_matches에 추가
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

# 좋은 매칭들을 거리 기준으로 정렬
# distance가 작을수록 두 특징점이 더 비슷하다고 볼 수 있음
good_matches = sorted(good_matches, key=lambda x: x.distance)

# 너무 많은 매칭이 한꺼번에 보이면 복잡하므로 상위 50개만 선택
good_matches = good_matches[:50]

# drawMatches()를 사용하여 두 이미지 사이의 매칭 결과를 시각화
# flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 옵션은
# 매칭되지 않은 단일 특징점은 그리지 않고 매칭된 점들만 표시함
matched_img = cv.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# drawMatches() 결과는 BGR 형식이므로
# matplotlib 출력을 위해 RGB 형식으로 변환
matched_img_rgb = cv.cvtColor(matched_img, cv.COLOR_BGR2RGB)

# 출력 창 크기를 설정
plt.figure(figsize=(16, 8))

# 매칭 결과 이미지를 출력
plt.imshow(matched_img_rgb)

# 그래프 제목을 설정
plt.title('SIFT Feature Matching')

# 축 눈금을 숨김
plt.axis('off')

# 그래프 간격을 자동으로 정리
plt.tight_layout()

# 최종 결과를 화면에 출력
plt.show()