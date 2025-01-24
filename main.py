import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import tiktoken
import time

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=6000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_length + sentence_tokens > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

def generate_minutes_from_chunks(client, transcript, title, description):
    chunks = chunk_text(transcript)
    minutes_sections = []
    
    for i, chunk in enumerate(chunks):
        time.sleep(3)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an AI assistant specialized in transforming meeting transcripts into structured meeting minutes. Extract key topics, discussions, recommendations, and action items."""},
                    {"role": "user", "content": f"Meeting Title: {title}\nDescription: {description}\nPart {i+1}/{len(chunks)}\n\nTranscript:\n{chunk}"}
                ]
            )
            minutes_sections.append(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error in chunk {i+1}: {str(e)}")
            time.sleep(60)
            continue
            
    return "\n\n".join(minutes_sections)

def split_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    chunk_duration = 5 * 60 * 1000
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

st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")

if 'step' not in st.session_state:
    st.session_state.step = 1

with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)

st.title("Meeting Minutes Generator")

if st.session_state.step == 1:
    st.subheader("Step 1: Meeting Details")
    title = st.text_input("Meeting Title")
    description = st.text_area("Meeting Description")
    if st.button("Next") and title and description:
        st.session_state.title = title
        st.session_state.description = description
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Step 2: Upload Audio")
    uploaded_file = st.file_uploader("Upload meeting recording", type=['mp3', 'wav', 'm4a'])
    
    if uploaded_file and api_key:
        with st.spinner("Processing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name

            try:
                chunks = split_audio(audio_path)
                transcript = transcribe_chunks(client, chunks)
                st.session_state.transcript = transcript
                os.unlink(audio_path)
                
                minutes = generate_minutes_from_chunks(client, transcript, st.session_state.title, st.session_state.description)
                st.session_state.minutes = minutes
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif st.session_state.step == 3:
    st.subheader("Generated Meeting Minutes")
    minutes = st.text_area("Meeting Minutes", st.session_state.minutes, height=400)
    st.session_state.minutes = minutes
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Minutes", st.session_state.minutes, "meeting_minutes.txt")
    with col2:
        st.download_button("Download Transcript", st.session_state.transcript, "transcript.txt")
    
    if st.button("Start New Meeting"):
        st.session_state.clear()
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.rerun()