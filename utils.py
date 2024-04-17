import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Function to load the pre-trained TensorFlow Lite model
def load_model(model_path):
    """
    Load a pre-trained TensorFlow Lite model.

    Args:
    - model_path: Path to the saved model file.

    Returns:
    - interpreter: Loaded TensorFlow Lite model interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess input image
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an input image for prediction.

    Args:
    - image_path: Path to the input image file.
    - target_size: Tuple specifying the target size for resizing the image.

    Returns:
    - preprocessed_image: Preprocessed image as a numpy array.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Resize the image
    image = image.resize(target_size)
    # Convert the image to a numpy array
    preprocessed_image = np.array(image) / 255.0  # Normalize pixel values
    return preprocessed_image

# Function to make predictions using the loaded model
def predict_grade(model, input_image):
    """
    Make predictions using the loaded model.

    Args:
    - model: Loaded TensorFlow Lite model interpreter.
    - input_image: Preprocessed input image for prediction.

    Returns:
    - predicted_class: Predicted class label for the input image.
    """
    class_labels = ['grade 1 quality 1', 'grade 1 quality 2', 'grade 1 quality 3', 'grade 1 quality 4',
                   'grade 2 quality 1', 'grade 2 quality 2', 'grade 2 quality 3', 'grade 2 quality 4',
                   'grade 3 quality 1', 'grade 3 quality 2', 'grade 3 quality 3', 'grade 3 quality 4']

    # Reshape input image to match model input shape
    input_details = model.get_input_details()
    input_shape = input_details[0]['shape']
    input_image = np.expand_dims(input_image, axis=0).astype(input_details[0]['dtype'])
    # Set input tensor
    model.set_tensor(input_details[0]['index'], input_image)
    # Run inference
    model.invoke()
    # Get output tensor
    output_details = model.get_output_details()
    output_data = model.get_tensor(output_details[0]['index'])
    # Find the index of the highest probability
    predicted_class_index = np.argmax(output_data)
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

def predict_disease(model, input_image):
    """
    Make predictions using the loaded model.

    Args:
    - model: Loaded TensorFlow Lite model interpreter.
    - input_image: Preprocessed input image for prediction.

    Returns:
    - predicted_class: Predicted class label for the input image.
    """
    class_labels = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

    # Reshape input image to match model input shape
    input_details = model.get_input_details()
    input_shape = input_details[0]['shape']
    input_image = np.expand_dims(input_image, axis=0).astype(input_details[0]['dtype'])
    # Set input tensor
    model.set_tensor(input_details[0]['index'], input_image)
    # Run inference
    model.invoke()
    # Get output tensor
    output_details = model.get_output_details()
    output_data = model.get_tensor(output_details[0]['index'])
    # Find the index of the highest probability
    predicted_class_index = np.argmax(output_data)
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

if __name__ == "__main__":
    disease_model = load_model("models/inception_fine_tune.tflite")
    # grade_model = load_model("models/pomogranate_grading.tflite")
    img = preprocess_image("imgs/pomogranate.jpeg")
    
    disease = predict_disease(disease_model, img)
    # grade = predict_grade(grade_model, img)
    
    print("Predicted Disease:", disease)
    # print("Predicted Grade:", grade)
