import streamlit as st
import librosa
import pandas as pd
import numpy as np
import joblib
import whisper
import os
from audio_recorder_streamlit import audio_recorder
from tensorflow.keras.models import load_model
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Speech Emotion Detection (RAVDESS)",
    layout="wide",
    page_icon="🎤"
)

# ==============================
# SESSION STATE
# ==============================
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🎙 Speech Emotion AI")

st.sidebar.markdown("### 😊 Supported Emotions")
st.sidebar.write(", ".join(joblib.load("model_american/label_encoder.pkl").classes_))

st.sidebar.caption("Built with Streamlit + Whisper + Deep Learning")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    model = load_model("model_american/classifier.keras")
    scaler = joblib.load("model_american/scaler.pkl")
    encoder = joblib.load("model_american/label_encoder.pkl")
    whisper_model = whisper.load_model("tiny")
    return model, scaler, encoder, whisper_model

with st.spinner("🚀 Loading AI Model..."):
    model, scaler, encoder, whisper_model = load_models()

EXPECTED_FEATURE_SIZE = scaler.mean_.shape[0]

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    if len(y) == 0:
        raise ValueError("Audio file is empty")

    target_length = 22050 * 3

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    features = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(rms.T, axis=0)
    ])

    if len(features) != EXPECTED_FEATURE_SIZE:
        raise ValueError(
            f"Feature size mismatch. Expected {EXPECTED_FEATURE_SIZE}, got {len(features)}"
        )

    return features.reshape(1, -1)

# ==============================
# MAIN UI
# ==============================
st.title("🎤 Speech Emotion Detection System")
st.subheader("Upload or Record Audio")

input_method = st.radio(
    "Select Input Type:",
    ["Upload Audio File", "Record Live Audio"]
)

# ==============================
# AUDIO INPUT
# ==============================
if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload WAV File",
        type=["wav"],
        key=f"uploader_{st.session_state.uploaded_file_key}"
    )

    if uploaded_file:
        st.session_state.audio_bytes = uploaded_file.read()

elif input_method == "Record Live Audio":
    recorded_audio = audio_recorder()
    if recorded_audio:
        st.session_state.audio_bytes = recorded_audio

# ==============================
# AUDIO PREVIEW
# ==============================
if st.session_state.audio_bytes:
    col1, col2 = st.columns([10, 1])

    with col1:
        st.audio(st.session_state.audio_bytes)

    with col2:
        if st.button("❌"):
            st.session_state.audio_bytes = None
            st.session_state.uploaded_file_key += 1
            st.rerun()

# ==============================
# PROCESS AUDIO
# ==============================
if st.session_state.audio_bytes:

    temp_path = "temp_audio.wav"

    try:
        with open(temp_path, "wb") as f:
            f.write(st.session_state.audio_bytes)

        with st.spinner("🔍 Analyzing Emotion..."):

            # Whisper transcription
            result = whisper_model.transcribe(
                temp_path,
                task="transcribe",
                fp16=False
            )

            transcript = result.get("text", "").strip()
            detected_language = result.get("language", "").upper()

            st.caption(f"Detected Language: {detected_language}")

            # Feature extraction
            features = extract_features(temp_path)
            features_scaled = scaler.transform(features)

            # 🔥 FIXED: NO DOUBLE SOFTMAX
            prediction = model.predict(features_scaled, verbose=0)[0]

            emotion_index = np.argmax(prediction)
            emotion = encoder.inverse_transform([emotion_index])[0]
            confidence = float(np.max(prediction) * 100)

            prob_df = pd.DataFrame({
                "Emotion": encoder.classes_,
                "Probability": prediction
            })

            emotion_emojis = {
                "angry": "😡",
                "calm": "😌",
                "disgust": "🤢",
                "fear": "😨",
                "happy": "😊",
                "neutral": "😐",
                "sad": "😢",
                "surprised": "😲"
            }
            emoji_icon = emotion_emojis.get(emotion.lower(), "")

            # ==============================
            # OUTPUT
            # ==============================
            st.success("✅ Prediction Complete!")

            left, right = st.columns([2, 1])

            with left:
                st.subheader("📝 Transcript")
                st.write(transcript if transcript else "No speech detected.")

                st.markdown(f"""
                <div style="padding:20px;border-radius:12px;
                background-color:#111827;color:white;">
                    <h2>{emoji_icon} {emotion.upper()}</h2>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 Emotion Distribution")

                fig = px.pie(
                    prob_df,
                    names="Emotion",
                    values="Probability",
                    hole=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.metric("Confidence", f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

st.markdown("---")
st.caption("RAVDESS Dataset + Regularized ANN Model")