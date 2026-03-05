# cv, np 모듈 임포트
import cv2 as cv                                  
import numpy as np                                

brush_size = 5                                    # 초기 붓 크기를 5로 설정
drawing = False                                   # 현재 드래그 상태인지 지정
color = (255, 0, 0)                               # 현재 붓 색상(BGR) 기본값: 파란색(255,0,0)

# 'soccer.jpg' 파일을 읽어서 img 변수에 저장
img = cv.imread('soccer.jpg')

def mouse_event(event, x, y, flags, param):        # 마우스 이벤트가 발생할 때마다 호출될 콜백 함수 정의
    global drawing, color, brush_size, img         # 함수 안에서 전역 변수 drawing,color, brush_size, img을 수정하기 위해 global 선언

    if event == cv.EVENT_LBUTTONDOWN:                  # 마우스 왼쪽 버튼을 누르면
        drawing = True                                 # 그리기 상태를 True로 변경(드래그 시작)
        color = (255, 0, 0)                            # 좌클릭은 파란색으로 설정
        cv.circle(img, (x, y), brush_size, color, -1)  # 클릭 위치에 현재 붓 크기 원을 채워서(-1) 그림 , -1 : 원 채움 유무 판단

    elif event == cv.EVENT_RBUTTONDOWN:                # 마우스 오른쪽 버튼을 누르면
        drawing = True                                 # 그리기 상태를 True로 변경(드래그 시작)
        color = (0, 0, 255)                            # 우클릭은 빨간색으로 설정
        cv.circle(img, (x, y), brush_size, color, -1)  # 클릭 위치에 현재 붓 크기 원을 채워서 그림

    elif event == cv.EVENT_MOUSEMOVE:                      # 마우스를 움직이는 이벤트
        if drawing:                                        # 현재 버튼을 누른 채(드래그) 그리는 중이라면
            cv.circle(img, (x, y), brush_size, color, -1)  # 이동한 위치마다 원을 찍어 연속적으로 그린 효과를 냄

    elif event == cv.EVENT_LBUTTONUP:              # 마우스 왼쪽 버튼을 떼는 순간
        drawing = False                            # 그리기 상태를 False로 변경(드래그 종료)

    elif event == cv.EVENT_RBUTTONUP:              # 마우스 오른쪽 버튼을 떼는 순간
        drawing = False                            # 그리기 상태를 False로 변경

cv.namedWindow("Paint")                            # "Paint"라는 이름의 창을 생성
cv.setMouseCallback("Paint", mouse_event)          # "Paint" 창에 마우스 콜백 함수를 연결

while True:                                        # 프로그램이 종료될 때까지 무한 반복
    cv.imshow("Paint", img)                        # 현재 캔버스 이미지를 "Paint" 창에 출력
    key = cv.waitKey(1)                            # 1ms 동안 키 입력을 대기

    if key == ord('+'):                            # '+' 키가 눌렸다면
        brush_size = min(15, brush_size + 1)       # 붓 크기를 1 증가시키되 최대 15를 넘지 않게 제한

    elif key == ord('-'):                          # '-' 키가 눌렸다면
        brush_size = max(1, brush_size - 1)        # 붓 크기를 1 감소시키되 최소 1보다 작아지지 않게 제한

    elif key == ord('q'):                          # 'q' 키가 눌렸다면
        break                                      # 무한 루프를 종료

cv.destroyAllWindows()                             # OpenCV로 띄운 모든 창을 닫음


