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