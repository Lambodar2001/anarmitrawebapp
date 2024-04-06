from flask import Flask, render_template, request, url_for, Response
import cv2
from ultralytics import YOLO
import cvzone
import tensorflow as tf
import numpy as np
from PIL import Image
import math

app = Flask(__name__)

# Set the prediction pipeline to use cpu only
tf.config.set_visible_devices([], 'GPU')

# Function to load the pre-trained TensorFlow model
def load_model(model_path):
    """
    Load a pre-trained TensorFlow model.

    Args:
    - model_path: Path to the saved model directory.

    Returns:
    - model: Loaded TensorFlow model.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Function to preprocess input image
def preprocess_image(image_path, target_size):
    """
    Preprocess an input image for prediction.

    Args:
    - image_path: Path to the input image file.
    - target_size: Tuple specifying the target size for resizing the image.

    Returns:
    - preprocessed_image: Preprocessed image as a numpy array.
    """
    # Load the image
    image = Image.open(image_path)
    # Resize the image
    image = image.resize(target_size)
    # Convert the image to a numpy array
    preprocessed_image = np.array(image)
    # Normalize pixel values (assuming input range [0, 255])
    return preprocessed_image

# Function to make predictions using the loaded model
def predict_grade(model, input_image):
    """
    Make predictions using the loaded model.

    Args:
    - model: Loaded TensorFlow model.
    - input_image: Preprocessed input image for prediction.

    Returns:
    - predicted_class: Predicted class label for the input image.
    """
    class_labels = ['grade 1 quality 1', 'grade 1 quality 2', 'grade 1 quality 3', 'grade 1 quality 4',
                   'grade 2 quality 1', 'grade 2 quality 2', 'grade 2 quality 3', 'grade 2 quality 4',
                   'grade 3 quality 1', 'grade 3 quality 2', 'grade 3 quality 3', 'grade 3 quality 4']

    # Reshape input image to match model input shape
    input_image = np.expand_dims(input_image, axis=0)
    # Make predictions
    predictions = model.predict(input_image)
    # Find the index of the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

def predict_disease(model, input_image):
    """
    Make predictions using the loaded model.

    Args:
    - model: Loaded TensorFlow model.
    - input_image: Preprocessed input image for prediction.

    Returns:
    - predicted_class: Predicted class label for the input image.
    """
    class_labels = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

    # Reshape input image to match model input shape
    input_image = np.expand_dims(input_image, axis=0)
    # Make predictions
    predictions = model.predict(input_image)
    # Find the index of the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

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
    image.resize((200, 200)).convert('RGB').save("static/uploads/custom_image.jpg")
    disease_model_path = 'models/inception_fine_tune.h5'
    disease_model = load_model(disease_model_path)
    grade_model_path = ('models/pomogranate_grading.h5')
    grade_model = load_model(grade_model_path)
    # Load and preprocess the input image
    # Example target size (adjust according to your model's input size)
    target_size = (256, 256)
    input_image = preprocess_image(uploaded_file, target_size)

    # Make predictions using the loaded models
    predicted_disease = predict_disease(disease_model, input_image)
    predicted_grade = predict_grade(grade_model, input_image)

    # Additional information about managing and curing diseases
    disease_info = {
        "Alternaria": "To manage Alternaria diseases in pomegranates, employing appropriate fungicides is crucial. Products based on copper oxychloride have shown high effectiveness. It is recommended to conduct two preventive sprayings during the blooming period or when initial symptoms manifest on the fruits. Additionally, fungicides containing propiconazole, thiophanate methyl, or azoxystrobin have proven to be highly effective. It's important to adhere to specified concentrations and rotate fungicides with different modes of action to prevent the development of resistance.",
        
        "Anthracnose": "Always consider an integrated approach with preventive measures together with biological treatments if available. A first preventive spray could be applied when flowering begins and the environmental conditions are favorable for the fungus. Then spray at 15-day intervals twice, if necessary. Active ingredients are propiconazole, mancozeb or a combination of mancozeb and tricyclazole. Only spray fungicides with an actual registration for pomegranate. It is important to follow the specified concentrations and to use fungicides with different mode of actions to prevent resistances.",
        
        "Bacterial_Blight": "For bacterial blight in pomegranates, adopt a multi-pronged approach for effective control. Utilize copper-based fungicides like copper oxychloride and apply them during the flowering period or at the onset of symptoms. Additionally, integrate biocontrol agents such as Bacillus subtilis, Pseudomonas fluorescence, and Trichoderma harzianum. Employ cultural methods like soaking Neem leaves in cow urine for pest and pathogen control. Follow up with applications of 40 Percentage Tulsi leaf extract followed by Neem seed oil. Incorporate an extract of Garlic bulb, Meswak stem, and Patchouli leaves at 30 Percentage concentration for added efficacy.",
        
        "Cercospora" : "Always consider an integrated approach with preventive measures together with biological treatments if available. If the economic threshold is reached, control measures have to be introduced. Two to three sprayings at 15 days interval of a fungicide after fruit formation gives good control of the disease. Active ingredients are mancozeb, conazole, or kitazin. Only spray fungicides with an actual registration for pomegranate. It is important to follow the specified concentrations and to use fungicides with different mode of actions to prevent resistances. To respect the waiting period is also very important.",
        
        "Healthy":"Healthy Pomegrante !!! To keep pomegranate plants healthy and disease-free, implement regular inspections, proper sanitation, and optimal growing conditions. Consider cultural practices like crop rotation, balanced fertilization, and biological treatments. If needed, use chemical control measures judiciously and rotate fungicides to prevent resistance. Monitor plants closely for any signs of disease and adjust management practices accordingly."

        
        
        
        
        
        # Add more diseases and their respective information here
    }

    # Get additional information based on the detected disease
    disease_additional_info = disease_info.get(predicted_disease, None)

    # Render the template with the predicted disease and additional information
    uploaded_image = url_for('static', filename='uploads/' + "custom_image.jpg")
    return render_template('detect.html', disease=predicted_disease, grade=predicted_grade, uploaded_image=uploaded_image, disease_additional_info=disease_additional_info)

# Initialize YOLO model
model = YOLO("./models/nano_best_10.pt")

# Class names for pomegranates
class_names = ["bud", "flower", "early-fruit", "mid-growth", "ripe"]

# Function to detect and display pomegranates
def detect_and_display(img):
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Using opencv
            # for bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)
            if conf > 0.7:    # Using cvzone
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # for class name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f"{class_names[cls]} {conf} ", (max(0, x1), max(35, y1)),
                                scale=1, thickness=1)
    return img

# Route for object detection from webcam
@app.route('/ripeness')
def detect_objects_webcam():
    return render_template('ripeness.html')

# Route for video feed from webcam
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = detect_and_display(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/stop_webcam')
def stop_webcam():
    # Redirect to a page displaying a message that the webcam feed has stopped
    return render_template('webcam_stopped.html')

if __name__ == '__main__':
    app.run()
