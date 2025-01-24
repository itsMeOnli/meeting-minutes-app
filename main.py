import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment

CHUNK_SIZE = 24 * 1024 * 1024
MAX_SINGLE_FILE = 25 * 1024 * 1024

def initialize_session_state():
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'cleaned_text' not in st.session_state:
        st.session_state.cleaned_text = None
    if 'meeting_minutes' not in st.session_state:
        st.session_state.meeting_minutes = None
    if 'step' not in st.session_state:
        st.session_state.step = 1

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

def clean_transcript(client, transcript, title, description):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """
            # IDENTITY and PURPOSE

            You are an expert at cleaning up broken and, malformatted, text, and formatting meeting transcripts. For example: line breaks in weird places, etc. 

            # Steps

            - Read the entire document and fully understand it.
            - Remove any strange line breaks that disrupt formatting.
            - Add capitalization, punctuation, line breaks, paragraphs and other formatting where necessary.
            - Do NOT change any content or spelling whatsoever.

            # OUTPUT INSTRUCTIONS

            - Output the full, properly-formatted text.
            - Do not output warnings or notes—just the requested sections.

            # INPUT:

            INPUT:
            """},
            {"role": "user", "content": f"Title: {title}\nDescription: {description}\n\nTranscript to clean:\n{transcript}"}
        ]
    )
    return response.choices[0].message.content

def generate_minutes(client, cleaned_text, title, description):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 
            """
            Create meeting minutes following this template:
            # Meeting Title
            Date: [Date]
            
            ## Attendees
            [List of attendees mentioned in the discussion]
            
            ## Agenda Items Discussed
            1. [Main topics discussed]
            
            ## Each 
            * [Important decisions made]
            
            ## Action Items
            * [Tasks assigned, deadlines, responsible parties]
            
            ## Next Steps
            * [Follow-up actions and future plans]"""},
            {"role": "user", "content": f"Title: {title}\nDescription: {description}\n\nCleaned transcript:\n{cleaned_text}"}
        ]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")
initialize_session_state()

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
        st.experimental_rerun()

elif st.session_state.step == 2:
    st.subheader("Step 2: Upload Audio")
    uploaded_file = st.file_uploader("Upload meeting recording", type=['mp3', 'wav', 'm4a'])
    
    if uploaded_file and api_key:
        file_size = len(uploaded_file.getvalue())
        with st.spinner("Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name

            try:
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
                
                st.session_state.transcription = transcript_text
                os.unlink(audio_path)
                st.session_state.step = 3
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")

elif st.session_state.step == 3:
    st.subheader("Step 3: Review Transcription")
    st.text_area("Transcription", st.session_state.transcription, height=300)
    st.download_button("Download Transcription", st.session_state.transcription, "transcription.txt")
    
    if st.button("Clean Transcript"):
        with st.spinner("Cleaning transcript..."):
            st.session_state.cleaned_text = clean_transcript(
                client, 
                st.session_state.transcription,
                st.session_state.title,
                st.session_state.description
            )
        st.session_state.step = 4
        st.experimental_rerun()

elif st.session_state.step == 4:
    st.subheader("Step 4: Review Cleaned Text")
    cleaned_text = st.text_area("Cleaned Text", st.session_state.cleaned_text, height=300)
    st.session_state.cleaned_text = cleaned_text
    st.download_button("Download Cleaned Text", st.session_state.cleaned_text, "cleaned_transcript.txt")
    
    if st.button("Generate Minutes"):
        with st.spinner("Generating meeting minutes..."):
            st.session_state.meeting_minutes = generate_minutes(
                client,
                st.session_state.cleaned_text,
                st.session_state.title,
                st.session_state.description
            )
        st.session_state.step = 5
        st.experimental_rerun()

elif st.session_state.step == 5:
    st.subheader("Step 5: Meeting Minutes")
    minutes = st.text_area("Meeting Minutes", st.session_state.meeting_minutes, height=400)
    st.session_state.meeting_minutes = minutes
    st.download_button("Download Meeting Minutes", st.session_state.meeting_minutes, "meeting_minutes.txt")
    
    if st.button("Start New Meeting"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.experimental_rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.experimental_rerun()