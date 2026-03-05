# cv, os, sys, np 모듈, datetime import
import cv2 as cv                      
import numpy as np                    
import os                         
import sys    
from datetime import datetime         

dragging = False                      # 드래그 중인지 여부
x0, y0 = -1, -1                       # 드래그 시작점 좌표
x1, y1 = -1, -1                       # 드래그 현재 좌표
roi = None                            # 선택된 ROI를 저장할 변수
has_roi = False                       # 유효한 ROI가 있는지 여부

# 'soccer.jpg' 파일을 읽어서 img1 변수에 저장
img = cv.imread('soccer.jpg')

# 이미지를 불러오지 못했을 경우 에러 메시지를 출력하고 프로그램 종료
if img is None:                       
    sys.exit("파일을 찾을 수 없습니다.")  

base = img.copy()                     # 원본 보존용(리셋 시 사용)
disp = img.copy()                     # 화면에 보여줄(사각형 그려질) 이미지

# =========================
# 좌표 정규화 함수
# - 드래그 방향이 어떤 방향이든(좌상->우하, 우하->좌상 등)
#   항상 (xmin,ymin)-(xmax,ymax)로 맞춰줌
# =========================
def normalize_rect(ax, ay, bx, by, w, h):    # 시작/끝 좌표 + 이미지 크기 입력
    x_min = max(0, min(ax, bx))              # x 최소(0 이상)
    y_min = max(0, min(ay, by))              # y 최소(0 이상)
    x_max = min(w - 1, max(ax, bx))          # x 최대(가로 범위 내)
    y_max = min(h - 1, max(ay, by))          # y 최대(세로 범위 내)
    return x_min, y_min, x_max, y_max        # 정규화된 사각형 좌표 반환

def on_mouse(event, x, y, flags, param):     # OpenCV가 마우스 이벤트 발생 시 호출
    global dragging, x0, y0, x1, y1          # 전역 상태 사용
    global disp, roi, has_roi, base          # 전역 이미지/ROI 상태 사용

    h, w = base.shape[:2]                    # 원본 이미지의 높이/너비 얻기

    if event == cv.EVENT_LBUTTONDOWN:        # 왼쪽 버튼을 누르면
        dragging = True                      # 드래그 시작 상태로 전환
        has_roi = False                      # 새로 선택하므로 이전 ROI 무효화
        roi = None                           # ROI 초기화
        x0, y0 = x, y                        # 시작점 저장
        x1, y1 = x, y                        # 현재점도 일단 시작점으로 지정
        disp = base.copy()                   # 화면 표시용 이미지를 원본으로 리셋

    elif event == cv.EVENT_MOUSEMOVE:        # 마우스를 움직이면
        if dragging:                         # 드래그 중일 때만
            x1, y1 = x, y                    # 현재 좌표 갱신
            disp = base.copy()               # 매 프레임 원본 복사
            # 사각형 좌표를 정규화(어느 방향 드래그든 처리)
            rx0, ry0, rx1, ry1 = normalize_rect(x0, y0, x1, y1, w, h)
            cv.rectangle(disp,                          # 표시 이미지 위에
                         (rx0, ry0),(rx1, ry1),         # 사각형 시작점, 사각형 끝점 
                         (0, 255, 0),2)                 # 초록색 두께 2로 사각형 그리기

    elif event == cv.EVENT_LBUTTONUP:        # 왼쪽 버튼을 떼면(드래그 종료)
        if dragging:                         # 드래그 중이었다면
            dragging = False                 # 드래그 종료 상태로 전환
            x1, y1 = x, y                    # 종료 좌표 저장
            # 사각형 좌표 정규화
            rx0, ry0, rx1, ry1 = normalize_rect(x0, y0, x1, y1, w, h)

            roi = base[ry0:ry1, rx0:rx1].copy()  # 잘라낸 ROI 복사
            has_roi = True                       # ROI가 유효함 표시

            disp = base.copy()                                 # 표시 이미지 초기화
            cv.rectangle(disp, (rx0, ry0), (rx1, ry1),         # 사각형 시작, 사각형 끝
                        (0, 255, 0), 2)                        # 초록색 두께 2

            cv.imshow("ROI", roi)            # 별도 창에 ROI 출력

# =========================
# 창 생성 및 콜백 등록
# =========================
cv.namedWindow("Image")                     # 메인 창 생성
cv.setMouseCallback("Image", on_mouse)      # 마우스 이벤트 콜백 등록

# =========================
# 메인 루프(키 입력 처리)
# =========================
while True:                                 
    cv.imshow("Image", disp)                # 현재 표시 이미지 출력(사각형 포함)
    key = cv.waitKey(1)                     # 키 입력(1ms 대기)

    if key == ord('q'):                     # q 키면 종료
        break                               # 루프 탈출

    elif key == ord('r'):                   # r 키면 리셋
        dragging = False                    # 드래그 상태 해제
        x0, y0, x1, y1 = -1, -1, -1, -1     # 좌표 초기화
        roi = None                          # ROI 초기화
        has_roi = False                     # ROI 없음
        disp = base.copy()                  # 표시 이미지 원본으로
        
        try:
            cv.destroyWindow("ROI")         # ROI 창 닫기
        except:
            pass                            # ROI 창이 없으면 예외 무시

    elif key == ord('s'):                   # s 키면 저장
        if not has_roi or roi is None:      # ROI가 없으면 저장 불가
            print("저장할 ROI가 없습니다. 먼저 드래그로 ROI를 선택하세요.")
            continue                        

        save_dir = "saved_roi"              # 저장 폴더 
        os.makedirs(save_dir, exist_ok=True)# 폴더 없으면 생성

        # 파일명에 시간 넣어서 중복 방지
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간 문자열
        save_path = os.path.join(save_dir, f"roi_{ts}.png")  # 저장 경로 구성

        ok = cv.imwrite(save_path, roi)     # ROI를 파일로 저장
        if ok:                              # 저장 성공 시
            print(f"ROI 저장 완료: {save_path}")       # 성공 메시지
        else:                               # 저장 실패 시
            print("ROI 저장 실패")                      # 실패 메시지

# =========================
# 종료 처리
# =========================
cv.destroyAllWindows()                       # 모든 창 닫기
