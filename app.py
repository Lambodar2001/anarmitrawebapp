from flask import Flask, render_template, request, url_for, Response
import cv2
from ultralytics import YOLO
import cvzone
import tensorflow as tf
from PIL import Image
import math
from utils import predict_disease, preprocess_image, load_model, predict_grade
app = Flask(__name__)

# Set the prediction pipeline to use cpu only
tf.config.set_visible_devices([], 'GPU')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def detect():
    return render_template('detect.html')


@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    uploaded_file = request.files['image']

    # Read the image file
    image = Image.open(uploaded_file)
    image.resize((200, 200)).convert('RGB').save(
        "static/uploads/custom_image.jpg")
    # disease_model = load_model("models/inception_fine_tune.tflite")
    grade_model = load_model("models/pomogranate_grading.tflite")
    img = preprocess_image("imgs/pomogranate.jpeg")

    # disease = predict_disease(disease_model, img)
    grade = predict_grade(grade_model, img)

    # print("Predicted Disease:", disease)
    print("Predicted Grade:", grade)
    # Load and preprocess the input image
    # Example target size (adjust according to your model's input size)
    target_size = (256, 256)
    input_image = preprocess_image(uploaded_file, target_size)

    # Make predictions using the loaded models
    # predicted_disease = predict_disease(disease_model, input_image)
    predicted_grade = predict_grade(grade_model, input_image)

    # Get additional information based on the detected disease
    # disease_additional_info = disease_info.get(predicted_disease, None)

    # Render the template with the predicted disease and additional information
    uploaded_image = url_for(
        'static', filename='uploads/' + "custom_image.jpg")
    return render_template('detect.html', grade=predicted_grade, uploaded_image=uploaded_image)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
