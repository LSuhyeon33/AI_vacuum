import tensorflow as tf
from keras.models import load_model

# load model
model = load_model('my_model.keras')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)
