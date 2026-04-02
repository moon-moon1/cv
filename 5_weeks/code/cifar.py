import os                                                       # 파일 존재 여부를 확인하기 위해 os를 불러옴
import numpy as np                                              # 배열 연산을 위해 numpy를 np로 불러옴
import tensorflow as tf                                         # TensorFlow 라이브러리를 tf로 불러옴
from tensorflow.keras.datasets import cifar10                   # CIFAR-10 데이터셋 로드를 위해 cifar10 모듈을 불러옴
from tensorflow.keras.models import Sequential                  # 순차형 모델 구성을 위해 Sequential을 불러옴
from tensorflow.keras.layers import Conv2D, MaxPooling2D        # CNN 구성을 위한 Conv2D, MaxPooling2D 레이어를 불러옴
from tensorflow.keras.layers import Flatten, Dense, Dropout     # 분류기를 위한 Flatten, Dense, Dropout 레이어를 불러옴
from tensorflow.keras.utils import load_img, img_to_array       # dog.jpg 로드와 배열 변환을 위해 유틸리티를 불러옴

# CIFAR-10 클래스 이름을 리스트로 정의
class_names = [
    'airplane',                                                 # 클래스 0
    'automobile',                                               # 클래스 1
    'bird',                                                     # 클래스 2
    'cat',                                                      # 클래스 3
    'deer',                                                     # 클래스 4
    'dog',                                                      # 클래스 5
    'frog',                                                     # 클래스 6
    'horse',                                                    # 클래스 7
    'ship',                                                     # 클래스 8
    'truck'                                                     # 클래스 9
]

# CIFAR-10 데이터셋을 로드
# x_train, y_train은 학습 데이터
# x_test, y_test는 테스트 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터셋의 형태를 출력
print("훈련 이미지 shape:", x_train.shape)                      #  (50000, 32, 32, 3)
print("훈련 라벨 shape:", y_train.shape)                        #  (50000, 1)
print("테스트 이미지 shape:", x_test.shape)                     #  (10000, 32, 32, 3)
print("테스트 라벨 shape:", y_test.shape)                       #  (10000, 1)

# 픽셀 값을 0~255 범위에서 0~1 범위로 정규화
# CNN 학습 시 수렴을 더 안정적으로 하기 위한 전처리
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 라벨은 (N, 1) 형태이므로 보기 편하게 1차원으로 바꿔도 되지만
# sparse_categorical_crossentropy는 현재 형태도 처리 가능하므로 그대로 사용 가능
# 여기서는 출력 편의를 위해 1차원으로 변환
y_train = y_train.flatten()
y_test = y_test.flatten()

# CNN 모델을 생성
model = Sequential()

# 첫 번째 합성곱 층을 추가
# 입력 이미지 크기는 32x32, 채널 수는 3(RGB)
# 필터 수는 32개, 커널 크기는 3x3, 활성화 함수는 ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# 첫 번째 풀링 층을 추가
# 2x2 최대 풀링으로 공간 크기를 줄여 계산량을 감소시킴
model.add(MaxPooling2D((2, 2)))

# 두 번째 합성곱 층을 추가
# 필터 수를 64개로 늘려 더 복잡한 특징을 추출
model.add(Conv2D(64, (3, 3), activation='relu'))

# 두 번째 풀링 층을 추가
model.add(MaxPooling2D((2, 2)))

# 세 번째 합성곱 층을 추가
# 필터 수를 다시 64개로 설정하여 특징을 더 정교하게 추출
model.add(Conv2D(64, (3, 3), activation='relu'))

# 3차원 특징맵을 1차원 벡터로 펼침
model.add(Flatten())

# 완전연결층(Dense)을 추가
# 은닉 노드 수는 64개, 활성화 함수는 ReLU
model.add(Dense(64, activation='relu'))

# 과적합 완화를 위해 Dropout을 추가
# 학습 시 노드의 30%를 무작위로 끔
model.add(Dropout(0.3))

# 출력층을 추가
# CIFAR-10은 10개 클래스를 가지므로 출력 노드 수는 10개
# softmax를 사용하여 각 클래스에 대한 확률을 출력
model.add(Dense(10, activation='softmax'))

# 모델의 학습 설정을 정의
# optimizer='adam'은 가중치 갱신 알고리즘
# loss='sparse_categorical_crossentropy'는 정수형 클래스 라벨에 적합한 손실 함수
# metrics=['accuracy']는 정확도를 함께 계산한다는 의미
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조를 출력
model.summary()

# 모델을 학습
# validation_split=0.1은 훈련 데이터의 10%를 검증용으로 사용
history = model.fit(
    x_train,                                                    # 학습 이미지
    y_train,                                                    # 학습 라벨
    epochs=30,                                                  # 전체 데이터를 10번 반복 학습
    batch_size=64,                                              # 한 번에 64장씩 학습
    validation_split=0.1                                        # 검증용 데이터 비율
)

# 테스트 데이터로 모델 성능을 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 테스트 손실값과 정확도를 출력
print(f"테스트 손실값: {test_loss:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")

# dog.jpg 파일이 존재하는지 확인
if os.path.exists("dog.jpg"):
    # dog.jpg 파일을 32x32 크기로 불러옴
    # CIFAR-10 입력 크기에 맞추기 위해 target_size를 (32, 32)로 설정
    dog_image = load_img("dog.jpg", target_size=(32, 32))

    # PIL 이미지를 NumPy 배열로 변환
    dog_array = img_to_array(dog_image)

    # 픽셀 값을 0~1 범위로 정규화
    dog_array = dog_array.astype("float32") / 255.0

    # 모델 입력 형태에 맞게 배치 차원(1장)을 추가
    dog_array = np.expand_dims(dog_array, axis=0)

    # dog.jpg에 대한 클래스 확률을 예측
    prediction = model.predict(dog_array, verbose=0)

    # 가장 확률이 높은 클래스 인덱스를 구함
    predicted_index = np.argmax(prediction[0])

    # 예측된 클래스 이름을 구함
    predicted_label = class_names[predicted_index]

    # 예측 확률을 구함
    predicted_prob = prediction[0][predicted_index]

    # 예측 결과를 출력
    print("\ndog.jpg 예측 결과")
    print("예측 클래스 인덱스:", predicted_index)
    print("예측 클래스 이름:", predicted_label)
    print(f"예측 확률: {predicted_prob:.4f}")

    # 전체 클래스별 확률도 함께 출력
    print("\n클래스별 예측 확률")
    for i, prob in enumerate(prediction[0]):
        print(f"{class_names[i]:>10s} : {prob:.4f}")
else:
    # dog.jpg가 현재 폴더에 없으면 안내 메시지를 출력
    print("\ndog.jpg 파일이 현재 작업 폴더에 없습니다.")
    print("같은 폴더에 dog.jpg를 넣고 다시 실행하면 예측 결과를 확인할 수 있습니다.")