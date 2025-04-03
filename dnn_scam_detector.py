import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from google.colab import drive

#  Mount Google Drive
drive.mount('/content/drive')

#  Dataset Paths
REAL_PATH = '/content/drive/MyDrive/ML_project_final_codes/otp_dataset/normal_calls/'
FAKE_PATH = '/content/drive/MyDrive/ML_project_final_codes/otp_dataset/scam_calls/'

#  Feature Extraction for DNN (Flattened MFCCs)
def extract_features_dnn(file_path, sr=16000, max_pad=128):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # 40 MFCCs for better representation
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_pad - mfcc.shape[1]))), mode='constant')  # Padding
    return mfcc[:, :max_pad].flatten()  # Flatten for DNN input

#  Load Dataset
def load_dataset():
    X, y = [], []

    # Load Normal Calls
    for file_name in os.listdir(REAL_PATH):
        file_path = os.path.join(REAL_PATH, file_name)
        if file_path.endswith('.wav'):
            features = extract_features_dnn(file_path)
            X.append(features)
            y.append(0)  # 0 for normal call

    # Load Scam Calls
    for file_name in os.listdir(FAKE_PATH):
        file_path = os.path.join(FAKE_PATH, file_name)
        if file_path.endswith('.wav'):
            features = extract_features_dnn(file_path)
            X.append(features)
            y.append(1)  # 1 for scam call

    return np.array(X), np.array(y)

#  Load Data
X, y = load_dataset()

#  Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#  Deep Neural Network (DNN) Model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(2, activation='softmax')  # Output layer for 2 classes
])

#  Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train Model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

#  Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f" Model Accuracy: {acc * 100:.2f}%")