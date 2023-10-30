import tensorflow as tf

# Create a TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")

# Allocate tensors and perform inference
interpreter.allocate_tensors()
