import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths
base_path = "/content/drive/MyDrive"
label_files = {
    'training-a': os.path.join(base_path, 'training-a', 'A-REFERENCE.csv'),
    'training-b': os.path.join(base_path, 'training-b', 'B-REFERENCE.csv'),
    'training-c': os.path.join(base_path, 'training-c', 'C-REFERENCE.csv'),
    'training-d': os.path.join(base_path, 'training-d', 'D-REFERENCE.csv'),
    'training-e': os.path.join(base_path, 'training-e', 'E-REFERENCE.csv'),
}

# Constants
TARGET_LENGTH = 10000
LOW_CUT = 20
HIGH_CUT = 800
SAMPLING_RATE = 2000
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512

def preprocess_signal(signal, sr):
    rms = np.sqrt(np.mean(signal**2))
    signal = signal / (rms + 1e-8)

    nyquist = 0.5 * sr
    low = LOW_CUT / nyquist
    high = HIGH_CUT / nyquist
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    signal = scipy_signal.filtfilt(b, a, signal)

    if len(signal) > TARGET_LENGTH:
        start = (len(signal) - TARGET_LENGTH) // 2
        signal = signal[start:start+TARGET_LENGTH]
    else:
        pad_length = TARGET_LENGTH - len(signal)
        signal = np.pad(signal, (0, pad_length), mode='reflect')

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        fmin=LOW_CUT,
        fmax=HIGH_CUT
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return features.T

# Load and preprocess data
X = []
y = []
for folder_name, label_path in label_files.items():
    dataset_path = os.path.join(base_path, folder_name)
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        continue

    try:
        df = pd.read_csv(label_path, header=None, names=['filename', 'label'])
        df['filename'] = df['filename'].str.strip()
        df = df[df['label'].isin([1, -1])]  # keep only valid labels

        for _, row in df.iterrows():
            fname = row['filename']
            label = 1 if row['label'] == 1 else 0  # 1=Abnormal, -1=Normal
            wav_file = os.path.join(dataset_path, fname + '.wav')
            if not os.path.exists(wav_file):
                print(f"File not found: {wav_file}")
                continue

            try:
                signal, sr = librosa.load(wav_file, sr=SAMPLING_RATE, mono=True)
                features = preprocess_signal(signal, sr)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
    except Exception as e:
        print(f"Error reading {label_path}: {str(e)}")

# Pad sequences to uniform length
max_length = max(x.shape[0] for x in X)
X = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')
X = np.array(X)
y = np.array(y)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Apply SMOTE on training set
n_samples, time_steps, n_features = X_train.shape
X_train_2D = X_train.reshape(n_samples, time_steps * n_features)
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_2D, y_train)
X_train_smote = X_train_smote.reshape(-1, time_steps, n_features)

# One-hot encode labels
y_train_smote = to_categorical(y_train_smote, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

# Add channel dimension for Conv2D
X_train_smote = X_train_smote[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN + LSTM model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(time_steps, n_features, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Reshape((int(time_steps/2), -1)),
    LSTM(64, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Train model (no early stopping)
history = model.fit(
    X_train_smote,
    y_train_smote,
    epochs=100,  # increased epochs for better training
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
    class_weight={0: 1., 1: 1.}
)

# Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Normal', 'Abnormal']))

cm = confusion_matrix(y_true, y_pred_classes)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Counts)')

plt.subplot(1, 2, 2)
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Percentages)')
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
