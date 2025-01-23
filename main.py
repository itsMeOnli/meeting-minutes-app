import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import math

CHUNK_SIZE = 24 * 1024 * 1024  # 24MB chunks
MAX_SINGLE_FILE = 25 * 1024 * 1024  # 25MB

st.set_page_config(page_title="Audio Transcriber")

def initialize_session_state():
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []

def split_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    chunk_duration = 5 * 60 * 1000  # 5 minutes in milliseconds
    chunks = []
    
    for i in range(0, duration, chunk_duration):
        chunk = audio[i:i + chunk_duration]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as chunk_file:
            chunk.export(chunk_file.name, format="mp3")
            chunks.append(chunk_file.name)
    
    return chunks

def transcribe_chunks(client, chunk_files):
    full_transcript = ""
    
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            full_transcript += transcript.text + " "
        os.unlink(chunk_file)
    
    return full_transcript.strip()

initialize_session_state()

with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)

st.title("Audio Transcriber")

uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'm4a'])

if uploaded_file and api_key:
    file_size = len(uploaded_file.getvalue())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    try:
        with st.spinner("Transcribing..."):
            if file_size > MAX_SINGLE_FILE:
                st.info(f"File size ({file_size/1024/1024:.1f}MB) exceeds 25MB. Splitting into chunks...")
                chunks = split_audio(audio_path)
                transcript_text = transcribe_chunks(client, chunks)
            else:
                with open(audio_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                transcript_text = transcript.text
            
            st.session_state.transcriptions.append({
                'filename': uploaded_file.name,
                'text': transcript_text
            })
        
        os.unlink(audio_path)
        
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")

if st.session_state.transcriptions:
    st.subheader("Transcriptions")
    for idx, trans in enumerate(st.session_state.transcriptions):
        with st.expander(f"Transcription {idx + 1}: {trans['filename']}"):
            st.write(trans['text'])
            
            st.download_button(
                label="Download as Text",
                data=trans['text'],
                file_name=f"{os.path.splitext(trans['filename'])[0]}_transcription.txt",
                mime="text/plain"
            )

if st.session_state.transcriptions:
    if st.button("Clear All Transcriptions"):
        st.session_state.transcriptions = []
        st.experimental_rerun()
