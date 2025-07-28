from google.colab import drive
drive.mount('/content/drive')

# Example: Access files in 'My Drive/myfolder'
folder_path = '/content/drive/MyDrive'

import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
import torch  # If you're planning to use PyTorch later
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Label files (all stored in their respective folders)
label_files = {
    'training-a': os.path.join(base_path, 'training-a', 'A-REFERENCE.csv'),
    'training-b': os.path.join(base_path, 'training-b', 'B-REFERENCE.csv'),
    'training-c': os.path.join(base_path, 'training-c', 'C-REFERENCE.csv'),
    'training-d': os.path.join(base_path, 'training-d', 'D-REFERENCE.csv'),
    'training-e': os.path.join(base_path, 'training-e', 'E-REFERENCE.csv'),
}

# Preprocessing parameters
TARGET_LENGTH = 10000  # 5s at 2000 Hz
LOW_CUT = 20
HIGH_CUT = 800
SAMPLING_RATE = 2000

X = []
y = []

def preprocess_signal(signal, sr):
    """Apply all preprocessing steps to a signal"""
    # 1. Normalization (min-max)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

    # 2. Bandpass filtering (20-800 Hz)
    nyquist = 0.5 * sr
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    signal = scipy_signal.filtfilt(b, a, signal)

    # 3. Segmentation to fixed length (5s = 10000 samples at 2000Hz)
    if len(signal) >= TARGET_LENGTH:
        signal = signal[:TARGET_LENGTH]
    else:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)))

    return signal

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

        label = 1 if label_raw == -1 else 0  # 1: Normal, 0: Abnormal
        wav_file = os.path.join(dataset_path, fname + '.wav')

        if not os.path.exists(wav_file):
            print(f"Missing file: {wav_file}")
            continue

        try:
            signal, sr = librosa.load(wav_file, sr=SAMPLING_RATE)
            signal = preprocess_signal(signal, sr)
            X.append(signal)
            y.append(label)
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")

# ✅ Convert lists to NumPy arrays after loading all data
X = np.array(X)
y = np.array(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Check shapes
print("Train set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

# Optional: Convert to torch tensors on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, 10000)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

# Reshape for CNN
X_train = X_train.reshape(-1, TARGET_LENGTH, 1)
X_val = X_val.reshape(-1, TARGET_LENGTH, 1)
X_test = X_test.reshape(-1, TARGET_LENGTH, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build CNN model
model = Sequential([
    Conv1D(64, kernel_size=100, activation='relu', input_shape=(TARGET_LENGTH, 1)),
    Conv1D(32, kernel_size=100, activation='relu'),
    Conv1D(32, kernel_size=100, activation='relu'),
    Dropout(0.5),
    MaxPooling1D(pool_size=2, strides=2),
    Flatten(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1)
# Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred_classes,
    labels=[0, 1],
    target_names=['Abnormal', 'Normal'],
    zero_division=0  # avoid divide-by-zero warnings
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Abnormal', 'Normal'],
            yticklabels=['Abnormal', 'Normal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (CNN Model)')
plt.show()

