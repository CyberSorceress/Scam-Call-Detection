import os
import librosa
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
train_folder = "/content/Training"
test_folder = "/content/Testing"
wav_folder = "/content/WAV"

# Ensure WAV folder exists
os.makedirs(wav_folder, exist_ok=True)

# Feature Extraction (Now accepts audio y and sr)
def extract_features(y, sr, window_size=10):
    duration = librosa.get_duration(y=y, sr=sr)
    features = []

    for start in range(0, int(duration), window_size):
        end = start + window_size
        if end > duration:
            break

        segment = y[start * sr:end * sr]  # Extract 10-sec window

        # MFSC + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # Additional Features
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y=segment)

        # Combine Features
        combined_features = np.hstack([
            np.mean(mfcc, axis=1), np.var(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1), np.var(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1), np.var(delta2_mfcc, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(chroma, axis=1),
            np.mean(zero_crossing)
        ])
        features.append(combined_features)

    return features if features else None

# Process all audio files
def process_audio(folder):
    X, y = [], []
    for file_name in os.listdir(folder):
        input_path = os.path.join(folder, file_name)

        # Convert to WAV if needed
        if file_name.endswith(".unknown") or file_name.endswith(".mp3"):
            output_path = os.path.join(wav_folder, file_name.rsplit(".", 1)[0] + ".wav")
            os.system(f'ffmpeg -i "{input_path}" -ar 16000 -ac 1 "{output_path}" -y')
            file_to_process = output_path
        elif file_name.endswith(".wav"):
            file_to_process = input_path
        else:
            continue

        # Load audio
        y_audio, sr = librosa.load(file_to_process, sr=16000)

        # Extract features
        features = extract_features(y_audio, sr)
        if features is not None:
            X.extend(features)
            label = 1 if "scam" in file_name.lower() else 0
            y.extend([label] * len(features))

            # Apply Data Augmentation for Scam Calls
            if label == 1:
                pitch_shifted = extract_features(librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=2), sr)
                time_stretched = extract_features(librosa.effects.time_stretch(y_audio, rate=0.9), sr)
                noise_added = extract_features(y_audio + 0.005 * np.random.randn(len(y_audio)), sr)

                if pitch_shifted:
                    X.extend(pitch_shifted)
                    y.extend([1] * len(pitch_shifted))
                if time_stretched:
                    X.extend(time_stretched)
                    y.extend([1] * len(time_stretched))
                if noise_added:
                    X.extend(noise_added)
                    y.extend([1] * len(noise_added))

    return np.array(X), np.array(y)

# Load Data
X_train, y_train = process_audio(train_folder)
X_test, y_test = process_audio(test_folder)

# Print training and testing sizes
print(f"Training Size: X_train = {X_train.shape[0]}, y_train = {y_train.shape[0]}")
print(f"Testing Size: X_test = {X_test.shape[0]}, y_test = {y_test.shape[0]}")

# Compute Class Weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"⚖ Class Weights: {class_weight_dict}")

# Train XGBoost with Class Weighting
model = XGBClassifier(n_estimators=250, max_depth=7, scale_pos_weight=class_weight_dict[1], random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" Model Accuracy: {accuracy * 100:.2f}%\n")
print(" Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Scam'], yticklabels=['Ham', 'Scam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Model
with open("scam_detector.pkl", "wb") as f:
    pickle.dump(model, f)