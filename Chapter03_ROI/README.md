# 마우스로 영역 선택 및 ROI(관심영역) 추출

이 프로젝트는 OpenCV를 활용하여 사용자가 마우스 드래그로 이미지의 특정 영역(ROI, Region of Interest)을 선택하고, 선택된 영역을 별도의 창으로 확인한 뒤 파일로 저장할 수 있는 프로그램입니다.

## 📌 주요 기능
* **자유로운 마우스 드래그**: 마우스를 좌상단에서 우하단으로 끌거나, 그 반대로 끌어도 에러가 발생하지 않도록 좌표 정규화(Normalization) 기능을 구현했습니다.
* **실시간 영역 표시**: 마우스를 드래그하는 동안 초록색 사각형(Bounding Box)이 실시간으로 그려집니다.
* **ROI 독립 창 출력**: 마우스 클릭을 해제하면 잘라낸 영역만 보여주는 독립된 "ROI" 창이 팝업됩니다.
* **스마트 저장 시스템**: `s` 키를 누르면 `saved_roi` 폴더를 자동으로 생성하고, 파일명이 겹치지 않도록 현재 시간을 기준(`roi_YYYYMMDD_HHMMSS.png`)으로 이미지를 저장합니다.
* **초기화(Reset)**: 선택을 실수했을 경우 `r` 키를 눌러 화면을 원본 상태로 즉시 되돌릴 수 있습니다.

## 🛠️ 요구 사항 (Prerequisites)
기본 내장 모듈(`os`, `sys`, `datetime`) 외에 아래 라이브러리 설치가 필요합니다.

```bash
pip install opencv-python numpy
```

## 🕹️ 조작 방법 (Controls)

| 조작 | 기능 설명 |
|---|---|
| **마우스 좌클릭 + 드래그** | 자르고 싶은 영역(ROI) 지정 |
| **마우스 좌클릭 해제** | 영역 선택 완료 및 ROI 팝업창 생성 |
| **키보드 `s`** | 선택한 ROI 영역을 이미지 파일로 저장 |
| **키보드 `r`** | 선택 영역 초기화 (리셋) |
| **키보드 `q`** | 프로그램 완전 종료 |

## 📝 주요 코드 설명

### 1. 드래그 방향 예외 처리 (좌표 정규화)
```python
def normalize_rect(ax, ay, bx, by, w, h):
    x_min = max(0, min(ax, bx))
    y_min = max(0, min(ay, by))
    # ... (생략) ...
    return x_min, y_min, x_max, y_max
```
> 📐 사용자가 마우스를 우하단에서 좌상단으로(역방향으로) 드래그하더라도, 항상 시작점과 끝점의 최소/최대 좌표를 계산하여 올바른 사각형 영역을 잡을 수 있도록 보정합니다. 또한, 화면 밖으로 마우스가 나가도 에러가 나지 않도록 이미지 크기(`w`, `h`) 내로 제한합니다.

### 2. ROI(관심 영역) 추출
```python
roi = base[ry0:ry1, rx0:rx1].copy()
```
> ✂️ 넘파이(NumPy) 배열 슬라이싱을 이용해 원본 이미지(`base`)에서 사용자가 지정한 좌표만큼의 픽셀 데이터만 쏙 잘라내어 `roi` 변수에 저장합니다.

### 3. 안전한 폴더 생성 및 타임스탬프 저장
```python
save_dir = "saved_roi"
os.makedirs(save_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"roi_{ts}.png")
```
> 💾 `os.makedirs`의 `exist_ok=True` 옵션을 통해 폴더가 없을 때만 생성하도록 안전하게 처리했습니다. 또한 덮어쓰기 방지를 위해 저장 순간의 시간을 파일명에 포함시킵니다.

## 💻 실행 화면

<img src="img/roi_result.png" width="700">
