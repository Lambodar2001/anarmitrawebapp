import tensorflow as tf

model = tf.keras.models.load_model("models/pomogranate_grading.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("models/pomogranate_grading.tflite", "wb").write(tflite_model)