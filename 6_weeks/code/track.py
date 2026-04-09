import os
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment # 헝가리안 알고리즘(최적 매칭)을 위한 라이브러리

# ==========================================
# 1. 사용자 설정 (Configuration)
# ==========================================
VIDEO_PATH = "slow_traffic_small.mp4"      # 분석할 입력 비디오 파일 경로
YOLO_CFG = "yolov3.cfg"                   # YOLOv3 네트워크 구조 설정 파일
YOLO_WEIGHTS = "yolov3.weights"           # YOLOv3 사전 학습된 가중치 파일
OUTPUT_PATH = "tracked_slow_traffic.mp4"  # 추적 결과가 그려진 비디오를 저장할 경로

# 객체 검출(Detection) 관련 임계값
CONF_THRESHOLD = 0.5    # 이 값 이상으로 확신하는(Confidence) 객체만 인정함 (0.5 = 50% 이상)
NMS_THRESHOLD = 0.4     # 겹치는 박스를 제거할 때 사용하는 IoU 기준값 (Non-Maximum Suppression)

# 객체 추적(SORT) 관련 파라미터
IOU_THRESHOLD = 0.3     # 이전 프레임의 추적 박스와 현재 프레임의 검출 박스가 최소 30%는 겹쳐야 동일 객체로 매칭함
MAX_AGE = 10            # 화면에서 객체가 사라져도(가려짐 등) 바로 삭제하지 않고 10프레임 동안은 추적 상태를 유지함(기다려줌)
MIN_HITS = 3            # 최소 3프레임 연속으로 검출되어야 "안정적인 추적 대상"으로 판단하고 화면에 ID를 표시함

# 추적할 특정 클래스 지정 (교통 체증 영상에 맞는 클래스들)
TRACK_ONLY = {"person", "bicycle", "car", "motorbike", "bus", "truck"}

# COCO 데이터셋의 80개 클래스 이름 (별도의 .names 파일 없이 코드에 직접 내장)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ==========================================
# 2. 유틸리티 함수: IoU (교차 영역 비율) 계산
# ==========================================
def iou_xyxy(box_a, box_b):
    """
    두 개의 Bounding Box(경계 상자)가 얼마나 겹치는지 비율(0~1)을 계산합니다.
    1에 가까울수록 두 박스가 완벽히 겹친다는 의미입니다.
    """
    # 겹치는 영역의 좌상단, 우하단 좌표 계산
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # 겹치는 영역의 너비와 높이 (겹치지 않으면 0)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h # 교집합 면적

    # 각 박스의 개별 면적
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    # 합집합 면적 = A면적 + B면적 - 교집합 면적
    union = area_a + area_b - inter_area
    
    if union <= 0:
        return 0.0
        
    return inter_area / union # IoU = 교집합 / 합집합


# ==========================================
# 3. 박스 좌표 <-> 칼만 필터 상태 변환 함수
# ==========================================
# SORT 알고리즘의 칼만 필터는 좌표를 [x1, y1, x2, y2]로 쓰지 않고,
# [중심x(cx), 중심y(cy), 면적(s), 가로세로비율(r)] 형태로 변환하여 추적합니다.

def bbox_to_z(bbox):
    """ [x1, y1, x2, y2] 박스를 칼만 필터 입력용 측정값 [cx, cy, s, r]로 변환 """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    
    cx = x1 + w / 2.0    # 중심 X
    cy = y1 + h / 2.0    # 중심 Y
    s = w * h            # 면적(Scale)
    r = w / h            # 종횡비(Aspect Ratio)
    
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)

def x_to_bbox(state):
    """ 칼만 필터의 예측 결과 [cx, cy, s, r]를 다시 그리기용 박스 [x1, y1, x2, y2]로 복원 """
    cx = float(state[0, 0])
    cy = float(state[1, 0])
    s = float(state[2, 0])
    r = float(state[3, 0])

    # 면적이나 비율이 0 이하면 유효하지 않은 박스
    if s <= 0 or r <= 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)

    # 수학적으로 w * h = s 이고, w / h = r 이므로
    # w^2 = s * r 을 통해 w와 h를 구함
    w = np.sqrt(s * r)
    h = s / w

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ==========================================
# 4. 개별 객체 추적기 (Kalman Filter Tracker)
# ==========================================
class KalmanBoxTracker:
    """ 객체 1개당 1개씩 생성되어 해당 객체의 미래 위치를 예측하고 상태를 기억하는 클래스 """
    count = 0 # 모든 객체가 고유한 ID를 갖도록 클래스 변수로 카운트 유지

    def __init__(self, bbox, class_id, score):
        # 7개의 상태 변수 [cx, cy, s, r, cx속도, cy속도, s속도]를 예측하고,
        # 4개의 측정 변수 [cx, cy, s, r]를 입력받는 칼만 필터 생성
        self.kf = cv.KalmanFilter(7, 4)

        # 상태 전이 행렬 (Transition Matrix)
        # 등속도 운동(Velocity) 모델: "현재 위치 = 이전 위치 + 속도"를 수학적 행렬로 표현
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0], # cx_new = cx + vx
            [0, 1, 0, 0, 0, 1, 0], # cy_new = cy + vy
            [0, 0, 1, 0, 0, 0, 1], # s_new = s + vs
            [0, 0, 0, 1, 0, 0, 0], # r_new = r (종횡비는 변하지 않는다고 가정)
            [0, 0, 0, 0, 1, 0, 0], # vx_new = vx (등속도)
            [0, 0, 0, 0, 0, 1, 0], # vy_new = vy
            [0, 0, 0, 0, 0, 0, 1], # vs_new = vs
        ], dtype=np.float32)

        # 측정 행렬 (Measurement Matrix)
        # 관측된 데이터 [cx, cy, s, r]를 상태 변수 7개와 매핑
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # 노이즈 공분산 설정 (작은 값일수록 모델을 더 신뢰함)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        
        # 첫 번째 프레임의 위치로 필터 상태 초기화
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4] = bbox_to_z(bbox)

        # 추적기 관리용 변수들
        self.time_since_update = 0 # 객체가 검출되지 않고 흘러간 프레임 수
        self.id = KalmanBoxTracker.count # 고유 ID 부여
        KalmanBoxTracker.count += 1
        self.hits = 1              # 총 매칭(업데이트) 횟수
        self.hit_streak = 1        # 최근 연속 매칭 횟수
        self.age = 0               # 트래커가 생성된 후 지난 총 프레임 수
        self.class_id = int(class_id)
        self.score = float(score)

    def predict(self):
        """ 다음 프레임에서 이 객체가 어디에 있을지 '예측'함 """
        # 면적(s)과 면적변화속도(vs)의 합이 0 이하가 되면 박스가 뒤집히므로 속도를 0으로 보정
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0

        predicted = self.kf.predict() # OpenCV 칼만 필터 예측 수행
        self.age += 1

        # 업데이트가 안 되고(검출 실패) 예측만 수행됐다면 연속 매칭 횟수 초기화
        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        return x_to_bbox(predicted[:4]) # 예측된 상태를 다시 박스 좌표로 반환

    def update(self, bbox, class_id, score):
        """ 실제 검출된 박스(측정값)로 칼만 필터의 상태를 '수정(보정)'함 """
        measurement = bbox_to_z(bbox)
        self.kf.correct(measurement) # 칼만 필터 보정 (예측값과 측정값의 융합)
        
        self.time_since_update = 0   # 다시 검출되었으므로 미검출 프레임 수 0으로 초기화
        self.hits += 1
        self.hit_streak += 1
        self.class_id = int(class_id)
        self.score = float(score)

    def get_state(self):
        """ 현재 추적기의 객체 위치 박스 반환 """
        return x_to_bbox(self.kf.statePost[:4])


# ==========================================
# 5. 데이터 연관 (Data Association) - 헝가리안 알고리즘
# ==========================================
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    현재 프레임에서 찾은 '새로운 검출 박스들'과 칼만 필터가 예측한 '기존 트래커 박스들'을
    서로 짝지어주는(매칭) 함수입니다.
    """
    # 트래커나 검출 결과가 아예 없다면 빈 배열 반환
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    # [검출 개수 x 트래커 개수] 크기의 비용 행렬(Cost Matrix) 생성 (여기서는 IoU 값을 저장)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    # 모든 검출 박스와 모든 트래커 박스 간의 IoU(겹침 정도)를 계산
    for d, det in enumerate(detections):
        det_box = det[:4]
        det_cls = int(det[5])
        for t, trk in enumerate(trackers):
            trk_box = trk.get_state()
            trk_cls = trk.class_id
            
            # 클래스가 다르면(예: 사람은 자동차와 매칭되면 안 됨) IoU를 0으로 처리
            if det_cls != trk_cls:
                iou_matrix[d, t] = 0.0
            else:
                iou_matrix[d, t] = iou_xyxy(det_box, trk_box)

    # 헝가리안 알고리즘 (Linear Sum Assignment)
    # scipy의 이 함수는 '최소 비용'을 찾는 함수입니다.
    # 우리는 IoU가 '최대'인 짝을 찾아야 하므로, iou_matrix에 마이너스(-)를 붙여서 넘겨줍니다.
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_indices = []
    unmatched_detections = []
    unmatched_trackers = []

    assigned_det = set()
    assigned_trk = set()

    # 헝가리안 알고리즘이 찾아준 최적의 짝(r, c)들을 확인
    for r, c in zip(row_ind, col_ind):
        # 최적의 짝이더라도 겹치는 비율이 임계값(0.3)보다 낮으면 엉뚱한 매칭으로 간주하고 무시
        if iou_matrix[r, c] >= iou_threshold:
            matched_indices.append([r, c]) # 성공적인 매칭
            assigned_det.add(r)
            assigned_trk.add(c)

    # 짝을 찾지 못한 '새로운 검출'들 (화면에 새로 나타난 객체일 가능성 높음)
    for d in range(len(detections)):
        if d not in assigned_det:
            unmatched_detections.append(d)

    # 짝을 찾지 못한 '기존 트래커'들 (객체가 화면 밖으로 나갔거나 가려졌을 가능성 높음)
    for t in range(len(trackers)):
        if t not in assigned_trk:
            unmatched_trackers.append(t)

    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.array(matched_indices, dtype=int)

    # 매칭된 쌍, 남은 검출, 남은 트래커를 반환
    return matched_indices, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


# ==========================================
# 6. YOLOv3 모델 로드 및 객체 검출 함수
# ==========================================
def load_yolo(cfg_path, weights_path):
    """ OpenCV의 DNN 모듈을 사용하여 YOLO 모델을 메모리에 올립니다. """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"YOLO cfg 파일이 없습니다: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights 파일이 없습니다: {weights_path}")

    # 다크넷(Darknet) 기반의 가중치와 설정 파일 로드
    net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # 백엔드 설정
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)     # CPU 연산 타겟 설정 (GPU 사용시 CUDA로 변경)

    layer_names = net.getLayerNames()
    # 결과를 출력하는 마지막 레이어들의 이름을 가져옴
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, COCO_CLASS_NAMES, output_layer_names


def detect_objects(frame, net, output_layer_names, class_names, conf_threshold=0.5, nms_threshold=0.4, track_only=None):
    """ 한 프레임 이미지에서 YOLO를 이용해 객체의 위치와 종류를 찾습니다. """
    h, w = frame.shape[:2]

    # 이미지를 YOLO 네트워크가 인식할 수 있는 블롭(Blob) 형태로 전처리 (크기 416x416 변환 및 정규화)
    blob = cv.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layer_names) # 네트워크 순방향 실행 (시간이 가장 오래 걸리는 부분)

    boxes = []
    confidences = []
    class_ids = []

    # 네트워크 출력값 해석
    for output in outputs:
        for detection in output:
            scores = detection[5:] # 처음 5개는 박스 정보, 나머지는 80개 클래스에 대한 확률값
            class_id = int(np.argmax(scores)) # 가장 확률이 높은 클래스 ID
            confidence = float(scores[class_id]) # 그 확률값(신뢰도)

            # 신뢰도가 임계값 미만이면 무시
            if confidence < conf_threshold:
                continue

            # 추적하고자 하는 클래스가 아니라면 무시 (예: 사람, 차만 추적)
            class_name = class_names[class_id]
            if track_only is not None and class_name not in track_only:
                continue

            # YOLO의 출력은 '박스의 중심 좌표'와 '너비/높이'의 상대 비율(0~1)입니다.
            # 이를 실제 이미지 픽셀 크기로 변환합니다.
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)

            x = int(center_x - bw / 2) # 좌상단 X
            y = int(center_y - bh / 2) # 좌상단 Y

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

    # NMS (Non-Maximum Suppression): 같은 객체에 여러 박스가 겹쳐서 그려지는 것을 방지(제거)
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            # 박스가 화면 밖으로 나가지 않도록 좌표를 잘라냄(Clip)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            detections.append([x1, y1, x2, y2, confidences[i], class_ids[i]])

    if len(detections) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(detections, dtype=np.float32) # [x1, y1, x2, y2, score, class_id] 형태의 배열 반환


# ==========================================
# 7. 시각화 보조 함수
# ==========================================
def color_for_id(track_id):
    """ 
    객체 ID마다 고유한 박스 색상을 지정해줍니다. 
    동일한 ID를 넣으면 항상 동일한 색상이 나옵니다 (난수 시드 고정).
    """
    np.random.seed(track_id + 12345)
    color = np.random.randint(0, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


# ==========================================
# 8. 메인 실행 함수 (프로그램의 심장부)
# ==========================================
def main():
    # 1. 모델 로드 및 비디오 캡처 객체 생성
    net, class_names, output_layer_names = load_yolo(YOLO_CFG, YOLO_WEIGHTS)

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {VIDEO_PATH}")

    # 영상 정보 가져오기 및 저장할 VideoWriter 설정
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    trackers = []    # 현재 추적 중인 객체들을 담을 리스트
    frame_count = 0  # 프레임 진행 번호

    # 2. 비디오 프레임 반복 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break # 영상이 끝나면 루프 탈출

        frame_count += 1

        # [단계 1] YOLO 모델을 통해 현재 프레임에 있는 객체들을 검출
        detections = detect_objects(
            frame, net, output_layer_names, class_names,
            conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD, track_only=TRACK_ONLY
        )

        # [단계 2] 기존 추적기(Kalman Filter)들을 한 프레임 미래로 '예측' (이동시킴)
        active_trackers = []
        for trk in trackers:
            predicted_box = trk.predict()
            # 예측 결과가 무한대(NaN) 등 비정상적인 값이 아니면 유지
            if not np.any(np.isnan(predicted_box)):
                active_trackers.append(trk)
        trackers = active_trackers

        # [단계 3] 검출된 객체(지금 본 것)와 추적기(예상되는 위치)를 헝가리안 알고리즘으로 짝짓기
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers, iou_threshold=IOU_THRESHOLD
        )

        # [단계 4] 짝이 맞는(매칭된) 기존 추적기는 검출된 위치 정보를 이용해 상태를 '보정(업데이트)'
        for det_idx, trk_idx in matched:
            det = detections[det_idx]
            bbox = det[:4]
            score = det[4]
            class_id = int(det[5])
            trackers[trk_idx].update(bbox, class_id, score)

        # [단계 5] 짝을 찾지 못한 '새로운 검출 박스'는 새로 화면에 들어온 객체이므로 '새 트래커'를 생성
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            bbox = det[:4]
            score = det[4]
            class_id = int(det[5])
            trackers.append(KalmanBoxTracker(bbox, class_id, score))

        # [단계 6] 트래커 관리 및 시각화 (화면 출력)
        survived_trackers = []
        for trk in trackers:
            # 설정한 MAX_AGE(10프레임) 이상 오랫동안 매칭되지 않은(검출 실패한) 트래커는 메모리에서 삭제
            if trk.time_since_update <= MAX_AGE:
                survived_trackers.append(trk)

            # 화면에 박스를 그릴 조건:
            # 1. 이번 프레임에서 제대로 검출되어 업데이트가 되었는가? (time_since_update == 0)
            # 2. 노이즈(오탐)가 아니라 최소 MIN_HITS(3회) 이상 안정적으로 잡힌 객체인가?
            should_draw = (trk.time_since_update == 0) and (
                trk.hits >= MIN_HITS or frame_count <= MIN_HITS
            )

            if should_draw:
                # 좌표를 정수형으로 변환 및 화면 밖으로 나가지 않게 클리핑
                x1, y1, x2, y2 = trk.get_state().astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width - 1, x2); y2 = min(height - 1, y2)

                color = color_for_id(trk.id) # 고유 색상
                class_name = class_names[trk.class_id]
                label = f"ID {trk.id} | {class_name} | {trk.score:.2f}"

                # 이미지 위에 사각형 박스와 정보 텍스트 그리기
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, max(20, y1 - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)

        # 살아남은 추적기들로 리스트 갱신
        trackers = survived_trackers

        # 좌측 상단에 프레임 수 표시
        cv.putText(frame, f"Frame: {frame_count}", (20, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

        # 결과 화면 실시간 표시 및 파일에 쓰기
        cv.imshow("YOLOv3 + SORT Multi-Object Tracking", frame)
        writer.write(frame)

        # 'q' 키를 누르면 강제 종료
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 3. 모든 작업이 끝나면 자원 해제(메모리 반환)
    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"추적 결과 저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()