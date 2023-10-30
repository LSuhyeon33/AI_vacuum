from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
import numpy as np
import cv2

# 데이터 불러오기
X_train = np.load('./x_train_data.npy')
X_test = np.load('./x_test_data.npy')
Y_train = np.load('./y_train_data.npy')
Y_test = np.load('./y_test_data.npy')

# X_train_samples = len(X_train)
# Y_train_samples = len(Y_train)
# X_test_samples = len(X_test)
# Y_test_samples = len(Y_test)

# print(f"X_train.shape : {X_train.shape}")
# print(f"Y_train.shape : {Y_train.shape}")
# print(f"X_test.shape : {X_test.shape}")
# print(f"Y_test.shape : {Y_test.shape}")

# print(f"Number of samples in X_train: {X_train_samples}")
# print(f"Number of samples in Y_train: {Y_train_samples}")
# print(f"Number of samples in X_test: {X_test_samples}")
# print(f"Number of samples in Y_test: {Y_test_samples}")

num_classes = 9

model = Sequential()
model.add(Convolution2D(16, (3, 3), padding='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Y_train과 Y_test가 원-핫 인코딩되어 있다고 가정합니다
model.fit(X_train, Y_train, batch_size=32, epochs=100,
          validation_data=(X_test, Y_test))

# test data에 대한 정확도 평가
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 모델 저장
model.save('my_model.keras')
