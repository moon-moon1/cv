# cv, sys, np 모듈 import
import cv2 as cv
import sys
import numpy as np

# 'soccer.jpg' 파일을 읽어서 img1 변수에 저장
img1 = cv.imread('soccer.jpg')

# 이미지를 불러오지 못했을 경우 에러 메시지를 출력하고 프로그램 종료
if img1 is None:
    sys.exit("파일을 찾을 수 없습니다.")

# 컬러(BGR) 이미지를 흑백(Gray) 이미지로 변환하여 img2에 저장
img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 흑백으로 변환된 이미지를 'soccer_gray.jpg'라는 이름의 파일로 저장
cv.imwrite('soccer_gray.jpg', img2)

# 흑백 이미지(1채널)를 다시 BGR(3채널) 형식으로 변환 (이미지 결합을 위해 채널 수를 맞춤)
img2_3channel = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

# 원본 이미지와 3채널 흑백 이미지를 가로로 이어 붙여 img3 생성
img3 = np.hstack((img1, img2_3channel))
    
# 'Image Display'라는 이름의 창을 띄우고 합쳐진 이미지(img3)를 화면에 표시
cv.imshow('Image Display', img3)

# 사용자가 키보드 아무 키나 누를 때까지 무한히 대기
cv.waitKey() 
cv.destroyAllWindows() # OpenCV로 띄운 모든 창을 닫음