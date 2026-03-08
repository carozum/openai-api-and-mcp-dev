import os
from colorama import Fore
import openai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  
client = openai.OpenAI()              

def save_file(text, file_name):
    """Saves content on a file"""
    
    try:
        # Ensure the `files` directory exists
        if not os.path.exists("temporary_files"):
            os.makedirs("temporary_files")
            

        # Open the file and write the content
        if text is None:
            text = ""
        else: 
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(text)
                print(f"Content saved to {file_name}")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def speech_to_text(audio_path):
    """Convert speech to text using OpenAI's Audio API
    https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create"""
    try:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        audio_file = open(audio_path, "rb")
        transcript = client.audio.transcriptions.create(
            # model="whisper-1",
            model="gpt-4o-transcribe",
            file=audio_file 
        )
        print(Fore.GREEN + f"Transcription successful: {transcript.text}" +  Fore.RESET  )
        return transcript.text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return ""  # return a string, not None


def speech_to_translation(audio_path):
    """ Translate speech audio into target language text 
    https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create"""
    try:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        audio_file = open(audio_path, "rb")
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file 
        )   
        print(Fore.GREEN + f"Translation successful: {translation.text}" +  Fore.RESET  )
        return translation.text   
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return ""  # return a string, not None   
    
def text_to_speech(
    text,
    output_path="temporary_files/audio.wav",
    model="tts-1",
    voice="alloy",
    instructions=None,
    response_format="wav",
    speed=1.0,
):
    """Convert text to speech and save to output_path
    https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create"""
    
    # Ensure directory exists
    if not os.path.exists("temporary_files"):
        os.makedirs("temporary_files")
        
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stream audio directly to file (recommended)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text[:4096],  # API limit per request
        instructions=instructions,
        response_format=response_format,
        speed=speed,
    ) as response:
        response.stream_to_file(out)

    return str(out)          