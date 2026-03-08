# Speech to text
# https://streamlit.io/components
# https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create

from colorama import Fore
import streamlit as st
from pathlib import Path    
import tempfile
from openai import OpenAI
import os
from dotenv import load_dotenv
from utils import speech_to_text, speech_to_translation, text_to_speech, save_file

# Load environment variables
load_dotenv()

client = OpenAI()

# Streamlit App
st.title("🔊 Audio transcriptions & Translations (Audio API)")  # Add a title

# Custom style 
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: transparent;
            border: 1px solid #3498db;
            float: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    textarea {
        color: #3498db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input
with st.form("user_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose a file")
    submit_button = st.form_submit_button(label="Submit")

# Process the uploaded file
if submit_button and uploaded_file is not None:
    with st.spinner("Transcribing..."):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
            print(f"Temporary file created at: {temp_file_path}")
            filename = temp_file_path.split("/")[-1]
            st.success("File processed successfully!")
            original_text = speech_to_text(temp_file_path)
            save_file(
                original_text,
                f"transcriptions/{filename}_transcription.txt",
            )
            st.markdown("Transcription:")
            st.audio(temp_file_path)
            st.markdown(f"<p style='color:#3498db'>{original_text}</p>", unsafe_allow_html=True)

            
    st.divider()        
    with st.spinner("Translating..."):  
        translated_text = speech_to_translation(temp_file_path)
        save_file(
                original_text,
                f"translations/{filename}_translation.txt",
        )
        text_to_speech(translated_text)
        st.markdown("Translation:")
        st.audio("temporary_files/audio.wav") 
        st.markdown(f"<p style='color:purple'>{translated_text}</p>", unsafe_allow_html=True)
        
#streamlit run main.py
