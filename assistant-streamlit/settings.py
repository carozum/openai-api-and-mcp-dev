import streamlit as st

# ----------------------------
# Model catalogs
# ----------------------------
TEXT_MODELS = {
    "GPT-4o Mini (fast)": "gpt-4o-mini",
    "GPT-4o (best quality)": "gpt-4o",
    "GPT-3.5 Turbo (legacy)": "gpt-3.5-turbo",
}

IMAGE_MODELS = {
    "GPT Image 1 (best quality)": "gpt-image-1",
    "GPT Image 1 Mini (cheaper)": "gpt-image-1-mini",
}

TTS_MODELS = {
    "TTS-1 (fast)": "tts-1",
    "TTS-1 HD (high quality)": "tts-1-hd",
}

TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

STT_MODELS = {
    "GPT-4o Transcribe (best)": "gpt-4o-transcribe",
    "Whisper-1 (legacy)": "whisper-1",
}


def setup_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")

        # --- Text ---
        st.subheader("💬 Text")
        text_label = st.selectbox("Text model", list(TEXT_MODELS.keys()), index=0)
        st.session_state["text_model"] = TEXT_MODELS[text_label]
        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

        # --- Image ---
        st.subheader("🖼️ Image")
        image_label = st.selectbox("Image model", list(IMAGE_MODELS.keys()), index=0)
        st.session_state["image_model"] = IMAGE_MODELS[image_label]

        # --- Audio ---
        st.subheader("🎙️ Audio")
        stt_label = st.selectbox("Transcription model", list(STT_MODELS.keys()), index=0)
        st.session_state["stt_model"] = STT_MODELS[stt_label]

        tts_label = st.selectbox("TTS model", list(TTS_MODELS.keys()), index=0)
        st.session_state["tts_model"] = TTS_MODELS[tts_label]

        st.session_state["tts_voice"] = st.selectbox("Voice", TTS_VOICES, index=0)
        st.session_state["tts_speed"] = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)