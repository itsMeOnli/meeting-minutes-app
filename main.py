import streamlit as st
import openai
import tempfile
import os

# Configure the OpenAI API key
st.set_page_config(page_title="Audio Transcriber")

def initialize_session_state():
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []

initialize_session_state()

# Sidebar for API key
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        openai.api_key = api_key

st.title("Audio Transcriber")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'm4a'])

if uploaded_file and api_key:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    try:
        with st.spinner("Transcribing..."):
            # Open the audio file and transcribe
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file
                )
            
            # Add transcription to session state
            st.session_state.transcriptions.append({
                'filename': uploaded_file.name,
                'text': transcript['text']
            })
            
        # Clean up temporary file
        os.unlink(audio_path)
        
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")

# Display transcriptions
if st.session_state.transcriptions:
    st.subheader("Transcriptions")
    for idx, trans in enumerate(st.session_state.transcriptions):
        with st.expander(f"Transcription {idx + 1}: {trans['filename']}"):
            st.write(trans['text'])
            
            # Download button for each transcription
            st.download_button(
                label="Download as Text",
                data=trans['text'],
                file_name=f"{os.path.splitext(trans['filename'])[0]}_transcription.txt",
                mime="text/plain"
            )

# Clear transcriptions button
if st.session_state.transcriptions:
    if st.button("Clear All Transcriptions"):
        st.session_state.transcriptions = []
        st.experimental_rerun()
