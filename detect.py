from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
import numpy as np
import pickle
from PIL import Image
from keras.models import load_model, Model
from sklearn.svm import SVC

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained CNN model
cnn_model = load_model('model/cnn_weights.hdf5')
cnn_feature_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Load the pre-trained SVM model
svm_model_file = 'model/svm_model.pkl'
try:
    with open(svm_model_file, 'rb') as f:
        svm_model = pickle.load(f)
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

labels = ['Body', 'Hand', 'Leg', 'None']

# Preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure it's in RGB format
    img = np.array(img)
    img = cv2.resize(img, (32, 32))  # Resize to CNN input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape for CNN model input
    return img

# Predict function
def predict_image(image):
    if svm_model is None:
        return "Error: SVM model not available."
    
    preprocessed_img = preprocess_image(image)
    cnn_features = cnn_feature_model.predict(preprocessed_img)
    
    try:
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(cnn_features)
            predicted_class = probabilities.argmax()
            if max(probabilities[0]) < 0.5:
                return "None"
        else:
            predicted_class = svm_model.predict(cnn_features)[0]
    except AttributeError:
        predicted_class = svm_model.predict(cnn_features)[0]
    
    return labels[predicted_class]

@app.route('/')
def home():
    return render_template('index.html', prediction=None, image_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file uploaded.', image_url=None)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction='No file selected.', image_url=None)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    image = Image.open(filepath)
    prediction = predict_image(image)
    
    return render_template('index.html', prediction=prediction, image_url=url_for('static', filename=f'uploads/{file.filename}'))

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
