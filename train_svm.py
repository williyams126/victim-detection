import pickle
from sklearn.svm import SVC
from keras.models import load_model, Model
import numpy as np
import os

# Load the trained CNN model
cnn_model = load_model('model/cnn_weights.hdf5')
cnn_features_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Load dataset
X = np.load('model/X.txt.npy')  # Loaded images array
Y = np.load('model/Y.txt.npy')  # Loaded labels array

# Normalize and preprocess images
X = X.astype('float32') / 255.0  # Normalize pixel values

# Extract CNN features
cnn_features = cnn_features_model.predict(X)

# Train SVM model
svm_cls = SVC(C=102.0, tol=1.9, probability=True)  # Enable probability
svm_cls.fit(cnn_features, Y)

# Save trained SVM model
with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_cls, f)

print("✅ SVM model trained successfully and saved as 'svm_model.pkl'.")
