# 컴퓨터 비전
본 저장소는 TensorFlow와 Keras를 이용한 손글씨 숫자 분류 과제를 정리한 페이지입니다.  
본 과제에서는 MNIST 데이터셋을 이용하여 간단한 이미지 분류기를 구현하였다.

---

# 1번 과제: MNIST 기반 간단한 이미지 분류기 구현

## 1. 문제 설명
손글씨 숫자 이미지 데이터셋인 MNIST를 이용하여 간단한 이미지 분류기를 구현하였다.  
MNIST는 28x28 크기의 흑백 손글씨 숫자 이미지로 구성되어 있으며, 각 이미지는 0부터 9까지의 숫자 중 하나의 클래스로 분류된다.  
본 과제에서는 MNIST 데이터를 불러온 뒤, 간단한 완전연결 신경망(MLP)을 구축하고 학습하여 테스트 데이터에 대한 분류 정확도를 평가하였다.

## 2. 요구사항
- MNIST 데이터셋을 로드
- 데이터를 훈련 세트와 테스트 세트로 분할
- 간단한 신경망 모델을 구축
- 모델을 훈련시키고 정확도를 평가

## 3. 사용한 주요 함수

### `mnist.load_data()`
MNIST 데이터셋을 불러오는 함수이다.  
훈련 데이터와 테스트 데이터를 자동으로 나누어 제공한다.

### `Sequential()`
신경망 레이어를 순차적으로 쌓아 모델을 구성하는 함수이다.

### `Flatten()`
2차원 이미지 데이터를 1차원 벡터로 변환하는 레이어이다.  
28x28 이미지를 784차원 벡터로 펼친다.

### `Dense()`
완전연결층(Fully Connected Layer)을 구성하는 레이어이다.  
은닉층과 출력층을 구현하는 데 사용하였다.

### `model.compile()`
손실 함수, 최적화 함수, 평가 지표를 설정하는 함수이다.

### `model.fit()`
훈련 데이터를 사용하여 모델을 학습시키는 함수이다.

### `model.evaluate()`
테스트 데이터를 사용하여 학습된 모델의 성능을 평가하는 함수이다.

## 4. 핵심 코드

### MNIST 데이터셋 로드
~~~python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
~~~

### 데이터 정규화
~~~python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
~~~

### 신경망 모델 구성
~~~python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
~~~

### 모델 학습
~~~python
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)
~~~

### 모델 평가
~~~python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
~~~

## 5. 중간 결과물

### 데이터셋 정보 확인
MNIST 데이터셋을 불러온 뒤, 훈련 이미지 수, 테스트 이미지 수, 이미지 크기를 출력하여 데이터가 정상적으로 로드되었는지 확인하였다.

```text
훈련 이미지 개수: 60000
테스트 이미지 개수: 10000
이미지 크기: (28, 28)
```

### 모델 구조 확인
`model.summary()`를 사용하여 입력층, 은닉층, 출력층으로 구성된 간단한 분류기 구조를 확인하였다.

## 6. 최종 결과물

### 테스트 정확도
학습이 끝난 뒤 테스트 데이터에 대해 손실값과 정확도를 출력하였다.

```text
테스트 손실값: 실행 환경에 따라 달라질 수 있음
테스트 정확도: 실행 환경에 따라 달라질 수 있음
```

### 예측 결과
테스트 이미지 일부에 대해 예측값과 실제 정답을 비교하여 출력하였다.

```text
1번째 이미지 -> 예측: ?, 실제: ?
2번째 이미지 -> 예측: ?, 실제: ?
...
```

### 해석
MNIST는 비교적 단순한 손글씨 숫자 데이터셋이므로, 복잡한 CNN이 아니더라도 간단한 완전연결 신경망만으로도 높은 정확도를 얻을 수 있다.  
또한 입력 픽셀 값을 0~1 범위로 정규화하면 학습이 더 안정적으로 진행된다.

## 7. 전체 코드

~~~python
import tensorflow as tf                                      # TensorFlow 라이브러리를 불러옴
from tensorflow.keras.datasets import mnist                  # MNIST 데이터셋을 불러오기 위한 모듈을 가져옴
from tensorflow.keras.models import Sequential               # 순차형 신경망 모델을 만들기 위한 Sequential을 가져옴
from tensorflow.keras.layers import Flatten, Dense           # 입력 펼치기용 Flatten, 완전연결층 Dense를 가져옴

# MNIST 데이터셋을 불러옴
# x_train, y_train은 훈련 데이터
# x_test, y_test는 테스트 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋의 기본 정보를 출력
print("훈련 이미지 개수:", x_train.shape[0])                  # 훈련 이미지 개수를 출력
print("테스트 이미지 개수:", x_test.shape[0])                  # 테스트 이미지 개수를 출력
print("이미지 크기:", x_train.shape[1:])                       # 각 이미지의 크기(28, 28)를 출력

# 픽셀 값을 0~255 정수 범위에서 0~1 실수 범위로 정규화
# 신경망 학습이 더 안정적으로 진행되도록 하기 위함
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 간단한 신경망 모델을 생성
model = Sequential([
    Flatten(input_shape=(28, 28)),                           # 28x28 이미지를 1차원 784 벡터로 펼침
    Dense(128, activation='relu'),                           # 은닉층 1: 뉴런 128개, 활성화 함수는 ReLU
    Dense(64, activation='relu'),                            # 은닉층 2: 뉴런 64개, 활성화 함수는 ReLU
    Dense(10, activation='softmax')                          # 출력층: 숫자 0~9 분류를 위해 10개 노드 사용
])

# 모델 학습 방식을 설정
# optimizer='adam'은 가중치 갱신 방법
# loss='sparse_categorical_crossentropy'는 정수형 레이블에 적합한 손실 함수
# metrics=['accuracy']는 정확도를 함께 계산하겠다는 의미
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조를 출력
model.summary()

# 모델을 훈련 데이터로 학습
# epochs=5는 전체 데이터를 5번 반복 학습한다는 의미
# batch_size=128은 한 번에 128장씩 묶어서 학습한다는 의미
# validation_split=0.1은 훈련 데이터의 10%를 검증용으로 사용한다는 의미
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 테스트 데이터로 모델 성능을 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 최종 테스트 손실값과 정확도를 출력
print("테스트 손실값:", test_loss)
print("테스트 정확도:", test_accuracy)

# 테스트 이미지 일부에 대해 예측 수행
predictions = model.predict(x_test[:10])

# 예측 결과와 실제 정답을 비교하여 출력
for i in range(10):
    predicted_label = tf.argmax(predictions[i]).numpy()      # 가장 확률이 높은 클래스를 예측값으로 선택
    true_label = y_test[i]                                   # 실제 정답 레이블을 가져옴
    print(f"{i+1}번째 이미지 -> 예측: {predicted_label}, 실제: {true_label}")
~~~

---

# 정리

1. MNIST 데이터셋 로드  
2. 데이터 정규화 및 간단한 신경망 구성  
3. 모델 학습 및 테스트 정확도 평가  
4. 테스트 이미지 예측 결과 확인


