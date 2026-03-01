import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ==============================
# CONFIG
# ==============================
DATASET_PATH = r"C:\Users\kovur\OneDrive\Desktop\vv"
SAMPLE_RATE = 22050
DURATION = 3
TARGET_LENGTH = SAMPLE_RATE * DURATION

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

# ==============================
# FEATURE EXTRACTION (UPGRADED)
# ==============================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(y) < TARGET_LENGTH:
            y = np.pad(y, (0, TARGET_LENGTH - len(y)))
        else:
            y = y[:TARGET_LENGTH]

        y = librosa.util.normalize(y)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(delta.T, axis=0),
            np.mean(delta2.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(rms.T, axis=0)
        ])

        return features

    except:
        return None


# ==============================
# LOAD DATA
# ==============================
X = []
y = []

print("Loading dataset...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav") and "-" in file:

            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code not in emotion_dict:
                continue

            emotion = emotion_dict[emotion_code]

            file_path = os.path.join(root, file)
            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Classes detected:", np.unique(y))
print("Total samples loaded:", len(X))

# ==============================
# ENCODING
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

os.makedirs("model_american", exist_ok=True)
joblib.dump(encoder, "model_american/label_encoder.pkl")

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "model_american/scaler.pkl")

# ==============================
# SPLIT
# ==============================
X_scaled, y_categorical = shuffle(X_scaled, y_categorical, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42
)

# ==============================
# CLASS WEIGHTS (IMPORTANT)
# ==============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))

# ==============================
# MODEL (STRONGER ANN)
# ==============================
model = Sequential()

model.add(Dense(1024, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(len(emotion_dict), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAIN
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# ==============================
# SAVE
# ==============================
model.save("model_american/classifier.keras")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")