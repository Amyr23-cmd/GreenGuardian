from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Specify the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load EfficientNet model
model_path = "CNN(Model3_EfficientNetB0).h5"
model = load_model(model_path)

# Class labels
class_labels = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites Two Spotted Spider Mite",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus"
]

# Function to preprocess the image before making predictions
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recdisease')
def recdisease():
    return render_template('recdisease.html')

@app.route('/abtus')
def abtus():
    return render_template('about us.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save the uploaded file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the uploaded image
    img_array = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(img_array)
    
    # Get all predictions regardless of confidence rate
    results = [{"class": class_labels[i], "confidence": float(prediction[0][i]) * 100} for i in range(len(class_labels))]
    
    # Filter out predictions with a confidence rate of 0
    results_filtered = [result for result in results if result["confidence"] > 0]

    # Return the predicted classes and confidence levels as JSON response
    return jsonify({"results": results_filtered})

if __name__ == '__main__':
    app.run(debug=True)
