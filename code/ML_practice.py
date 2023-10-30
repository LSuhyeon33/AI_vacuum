import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# 이번 시간 예시로 사용할 데이터를 불러오기 위한 라이브러리
from sklearn.datasets import load_digits
# ------------------------------------------------------------------------------------ #
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.optimizers import adadelta_v2, adam_v2, rmsprop_v2
from tensorflow.python.keras.utils import np_utils
import keras
# ------------------------------------------------------------------------------------ #
from keras import backend as K
# ------------------------------------------------------------------------------------ #

digits = load_digits()    # digits 이름으로 데이터 저장

x = digits.data
y = digits.target

# print("x =", x)
# print("y =", y)

# print(x.shape)
# print(y.shape)

plt.gray()                        # 그림을 흑백으로만 그림
plt.matshow(digits.images[0])    # 데이터를 매트릭스의 형태로 새로운 그림에 그려줌
# plt.show()

x_vars_stdscle = StandardScaler().fit_transform(x)
# train과 test 데이터로 나누기 (비율 = 0.7)
x_train, x_test, y_train, y_test = train_test_split(
    x_vars_stdscle, y, train_size=0.7, random_state=42)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# one-hot encoding
Y_train = np_utils.to_categorical(y_train, num_classes=10)

# print("[Before one-hot encoding]")
# print("y_trian =", y_train)
# print("[After one-hot encoding]")
# print("Y_train =", Y_train)

y_train_draw = pd.Series(y_train).value_counts()
# print(y_train_draw)
# print(y_train_draw.index)

plt.bar(y_train_draw.index, y_train_draw)
# plt.show()

np.random.seed(1337)
nb_classes = 10
batch_size = 128
nb_epochs = 200

# 첫번째 Hidden layer 지정
model = Sequential()
model.add(Dense(100, input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 두번째 Hidden layer 지정
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer 지정
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# learning rate 지정 전
model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=0)

# learning rate 지정 후
K.set_value(model.optimizer.learning_rate, 0.001)
model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=0)

y_train_predclass = np.argmax(model.predict(
    x_train, batch_size=batch_size), axis=1)

y_test_predclass = np.argmax(model.predict(
    x_test, batch_size=batch_size), axis=1)

model.predict(x_train, batch_size=batch_size)

print("\nDeep Neural Network - Train accuracy:\n\n",
      round(accuracy_score(y_train, y_train_predclass), 3))
print("\nDeep Neural Network - Train Classification Report\n\n",
      classification_report(y_train, y_train_predclass))
print("\nDeep Neural Network - Train Confusion Matrix\n\n", pd.crosstab(y_train,
      y_train_predclass, rownames=["Actual"], colnames=["Predicted"]))

print("\nDeep Neural Network - Test accuracy:\n\n",
      round(accuracy_score(y_test, y_test_predclass), 3))
print("\nDeep Neural Network - Test Classification Report\n\n",
      classification_report(y_test, y_test_predclass))
print("\nDeep Neural Network - Test Confusion Matrix\n\n", pd.crosstab(y_test,
      y_test_predclass, rownames=["Actual"], colnames=["Predicted"]))
