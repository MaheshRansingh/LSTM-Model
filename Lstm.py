import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Base path
base_path = "/content/drive/MyDrive"

# Label files
label_files = {
    'training-a': os.path.join(base_path, 'training-a', 'A-REFERENCE.csv'),
    'training-b': os.path.join(base_path, 'training-b', 'B-REFERENCE.csv'),
    'training-c': os.path.join(base_path, 'training-c', 'C-REFERENCE.csv'),
    'training-d': os.path.join(base_path, 'training-d', 'D-REFERENCE.csv'),
    'training-e': os.path.join(base_path, 'training-e', 'E-REFERENCE.csv'),
}

# Parameters
TARGET_LENGTH = 10000  # 5 seconds at 2000 Hz
LOW_CUT = 20  # Hz
HIGH_CUT = 800  # Hz
SAMPLING_RATE = 2000  # Hz
N_MFCC = 13  # MFCC features

# Preprocessing function
def preprocess_signal(signal, sr):
    # Normalize
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
    # Bandpass filter
    nyquist = 0.5 * sr
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    signal = scipy_signal.filtfilt(b, a, signal)
    # Pad or truncate
    if len(signal) >= TARGET_LENGTH:
        signal = signal[:TARGET_LENGTH]
    else:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)))
    # MFCC extraction
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)
    return mfcc

# Load and preprocess data
X = []
y = []

for folder_name, label_path in label_files.items():
    dataset_path = os.path.join(base_path, folder_name)
    if not os.path.exists(label_path):
        print(f"Label file missing: {label_path}")
        continue
    try:
        df = pd.read_csv(label_path, header=None, names=['filename', 'label'])
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        continue

    for _, row in df.iterrows():
        fname = row['filename'].strip()
        label_raw = row['label']
        if label_raw not in [1, -1]:
            continue
        label = 1 if label_raw == -1 else 0
        wav_file = os.path.join(dataset_path, fname + '.wav')
        if not os.path.exists(wav_file):
            print(f"Missing file: {wav_file}")
            continue
        try:
            signal, sr = librosa.load(wav_file, sr=SAMPLING_RATE)
            features = preprocess_signal(signal, sr)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

# Pad sequences to the longest MFCC sequence
X = pad_sequences(X, padding='post', dtype='float32')
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# SMOTE (flatten for 2D, then reshape back)
n_samples, time_steps, n_features = X_train.shape
X_train_2D = X_train.reshape((n_samples, time_steps * n_features))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_2D, y_train)
X_train_smote = X_train_smote.reshape(-1, time_steps, n_features)

# One-hot encode labels
y_train_smote = to_categorical(y_train_smote, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

# LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, n_features)),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train_smote, y_train_smote,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1)

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Abnormal', 'Normal']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Abnormal', 'Normal'],
            yticklabels=['Abnormal', 'Normal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (LSTM + MFCC + SMOTE)')
plt.show()

# Loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
