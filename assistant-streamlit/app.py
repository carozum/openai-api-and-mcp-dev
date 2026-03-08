import os
import streamlit as st
from utils import (
    generate_chat_completion,
    generate_chat_completion_with_history,
    generate_image,
    speech_to_text,
    speech_to_translation,
    text_to_speech,
)
from settings import setup_sidebar

st.title("Chatbot App")

setup_sidebar()

st.caption(
    f"Text: `{st.session_state.get('text_model', 'gpt-4o-mini')}` · "
    f"Image: `{st.session_state.get('image_model', 'gpt-image-1')}` · "
    f"Temp: `{st.session_state.get('temperature', 0.7)}` · "
    f"Voice: `{st.session_state.get('tts_voice', 'alloy')}`"
)

tab_chat, tab_image, tab_audio, tab_voice = st.tabs(
    ["💬 Text Chat", "🖼️ Image", "🎙️ Audio", "🤖 Voice Chat"]
)


# =================================================
# TAB — TEXT CHAT (multi-turn)
# =================================================

TMP_CHAT_AUDIO = "temporary_files/chat_voice_input.wav"

def _send_chat_message(user_text: str):
    """Shared logic to send a message and get a response."""
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    messages_for_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
    ]

    with st.spinner("..."):
        response = generate_chat_completion_with_history(
            messages_for_api,
            model=st.session_state.get("text_model", "gpt-4o-mini"),
            temperature=st.session_state.get("temperature", 0.7),
        )

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()


with tab_chat:

    # Init state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_transcription" not in st.session_state:
        st.session_state.chat_transcription = ""
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None

    # Display full conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Mic (hors form pour transcription immédiate) ──
    chat_audio = st.audio_input("🎤 Enregistre ta voix (optionnel)", key="chat_audio_input")

    if chat_audio is not None:
        # Utilise un hash du contenu pour détecter un nouvel enregistrement
        import hashlib
        audio_hash = hashlib.md5(chat_audio.getvalue()).hexdigest()
        if audio_hash != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_hash
            st.session_state.chat_transcription = ""  # reset pendant transcription
            try:
                os.makedirs("temporary_files", exist_ok=True)
                with open(TMP_CHAT_AUDIO, "wb") as f:
                    f.write(chat_audio.getbuffer())
                with st.spinner("🎧 Transcription..."):
                    st.session_state.chat_transcription = speech_to_text(
                        TMP_CHAT_AUDIO,
                        model=st.session_state.get("stt_model", "gpt-4o-transcribe"),
                    )
                st.rerun()
            except Exception as e:
                st.error(f"Erreur transcription : {e}")

    # ── Texte + bouton ──
    with st.form("chat_form", clear_on_submit=True):
        col_text, col_send = st.columns([8, 1.5])
        with col_text:
            user_input = st.text_input(
                "",
                value=st.session_state.chat_transcription,
                placeholder="Écris ton message...",
                label_visibility="collapsed",
                key="chat_text_input",
            )
        with col_send:
            submitted = st.form_submit_button("Envoyer", use_container_width=True, type="primary")

    if submitted and user_input:
        st.session_state.chat_transcription = ""
        st.session_state.last_audio_id = None
        _send_chat_message(user_input)

    # Clear button
    if st.session_state.chat_history:
        if st.button("🗑️ Effacer la conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# =================================================
# TAB — IMAGE
# =================================================

with tab_image:
    with st.form("image_form", clear_on_submit=True):
        prompt = st.text_input("Describe an image to generate")
        gen = st.form_submit_button("Generate image")

    if gen and prompt:
        with st.spinner("Generating..."):
            img = generate_image(prompt, model=st.session_state["image_model"])
        st.image(img, caption=prompt, use_container_width=True)


# =================================================
# TAB — AUDIO
# =================================================

TMP_AUDIO = "temporary_files/recorded.wav"

with tab_audio:

    # ── Section 1 : Speech → Text ──────────────────
    st.subheader("📝 Speech → Text")
    st.caption("Transcrit ta voix dans sa langue d'origine.")

    audio_stt = st.audio_input("Enregistre ta voix", key="stt_input")

    if audio_stt:
        os.makedirs("temporary_files", exist_ok=True)
        with open(TMP_AUDIO, "wb") as f:
            f.write(audio_stt.getbuffer())

        if st.button("📝 Transcrire", use_container_width=True):
            try:
                with st.spinner("Transcription en cours..."):
                    result = speech_to_text(
                        TMP_AUDIO,
                        model=st.session_state.get("stt_model", "gpt-4o-transcribe"),
                    )
                st.success(result)
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()

    # ── Section 2 : Speech → Translation ──────────
    st.subheader("🌐 Speech → Translation")
    st.caption("Traduit ta voix en anglais (utilise Whisper-1).")

    audio_trl = st.audio_input("Enregistre ta voix", key="trl_input")

    if audio_trl:
        os.makedirs("temporary_files", exist_ok=True)
        trl_path = "temporary_files/recorded_trl.wav"
        with open(trl_path, "wb") as f:
            f.write(audio_trl.getbuffer())

        if st.button("🌐 Traduire en anglais", use_container_width=True):
            try:
                with st.spinner("Traduction en cours..."):
                    result = speech_to_translation(trl_path)
                st.success(result)
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()

    # ── Section 3 : Text → Speech ──────────────────
    st.subheader("🔊 Text → Speech")
    st.caption("Synthétise un texte avec la voix choisie dans la sidebar.")

    tts_text = st.text_area(
        "Texte à synthétiser",
        placeholder="Écris quelque chose à entendre...",
        max_chars=4096,
    )

    if st.button("🎧 Générer l'audio", use_container_width=True) and tts_text:
        try:
            with st.spinner("Génération en cours..."):
                output_path = text_to_speech(
                    tts_text,
                    model=st.session_state.get("tts_model", "tts-1"),
                    voice=st.session_state.get("tts_voice", "alloy"),
                    speed=st.session_state.get("tts_speed", 1.0),
                )
            st.audio(output_path)
            st.caption(
                f"Modèle: `{st.session_state.get('tts_model')}` · "
                f"Voix: `{st.session_state.get('tts_voice')}` · "
                f"Vitesse: `{st.session_state.get('tts_speed')}`"
            )
        except Exception as e:
            st.error(f"Erreur : {e}")


# =================================================
# TAB — VOICE CHAT (micro uniquement, réponse audio, multi-turn en arrière-plan)
# =================================================

TMP_VOICE = "temporary_files/voice_input.wav"
TMP_RESPONSE = "temporary_files/voice_response.wav"

with tab_voice:

    # Init history (gardé en mémoire mais non affiché)
    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []

    st.caption("🎤 Enregistre ta question puis clique **Envoyer**.")

    voice_input = st.audio_input("", key="voice_chat_input")

    if voice_input:
        os.makedirs("temporary_files", exist_ok=True)
        with open(TMP_VOICE, "wb") as f:
            f.write(voice_input.getbuffer())

        if st.button("▶️ Envoyer", use_container_width=True, type="primary"):
            try:
                with st.spinner("🎧 Transcription..."):
                    user_text = speech_to_text(
                        TMP_VOICE,
                        model=st.session_state.get("stt_model", "gpt-4o-transcribe"),
                    )

                st.session_state.voice_history.append(
                    {"role": "user", "content": user_text}
                )

                messages_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.voice_history
                ]

                with st.spinner("🤔 Réflexion..."):
                    response_text = generate_chat_completion_with_history(
                        messages_for_api,
                        model=st.session_state.get("text_model", "gpt-4o-mini"),
                        temperature=st.session_state.get("temperature", 0.7),
                    )

                with st.spinner("🔊 Synthèse vocale..."):
                    audio_path = text_to_speech(
                        response_text,
                        output_path=TMP_RESPONSE,
                        model=st.session_state.get("tts_model", "tts-1"),
                        voice=st.session_state.get("tts_voice", "alloy"),
                        speed=st.session_state.get("tts_speed", 1.0),
                    )

                st.session_state.voice_history.append(
                    {"role": "assistant", "content": response_text}
                )

                # Lecture audio directe — autoplay via HTML
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                import base64
                audio_b64 = base64.b64encode(audio_bytes).decode()
                st.markdown(
                    f'<audio autoplay src="data:audio/wav;base64,{audio_b64}"></audio>',
                    unsafe_allow_html=True,
                )
                st.success("✅ Réponse envoyée !")

            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.voice_history:
        if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
            st.session_state.voice_history = []
            st.rerun()