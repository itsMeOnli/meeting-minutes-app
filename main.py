import streamlit as st
from openai import OpenAI
import tempfile
import os
from pydub import AudioSegment
import tiktoken

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

# Add token counting function
def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Add text chunking function
def chunk_text(text, max_tokens=30000):
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

def clean_transcript_in_chunks(client, transcript, title, description):
    chunks = chunk_text(transcript)
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
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
                {"role": "user", "content": f"Part {i+1}/{len(chunks)}\nTitle: {title}\nDescription: {description}\n\nTranscript chunk:\n{chunk}"}
            ]
        )
        cleaned_chunks.append(response.choices[0].message.content)
    
    return ' '.join(cleaned_chunks)

def clean_transcript_in_chunks(client, transcript, title, description):
    chunks = chunk_text(transcript)
    cleaned_chunks = []
    
    # First pass: Clean each chunk
    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", 
                "content": """
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
                """
                },
                {"role": "user", 
                "content": f"Part {i+1}/{len(chunks)}\nTitle: {title}\nDescription: {description}\n\nTranscript chunk:\n{chunk}"}
            ]
        )
        cleaned_chunks.append(response.choices[0].message.content)
    
    # Second pass: Summarize the cleaned chunks
    combined_clean = " ".join(cleaned_chunks)
    if count_tokens(combined_clean) > 30000:
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Create a concise summary of this meeting transcript, focusing on key points, decisions, and action items."},
                {"role": "user", "content": f"Title: {title}\nDescription: {description}\n\nTranscript:\n{combined_clean}"}
            ]
        )
        return final_response.choices[0].message.content
    
    return combined_clean

# def clean_transcript(client, transcript, title, description):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": """
#             # IDENTITY and PURPOSE

#             You are an expert at cleaning up broken and, malformatted, text, and formatting meeting transcripts. For example: line breaks in weird places, etc. 

#             # Steps

#             - Read the entire document and fully understand it.
#             - Remove any strange line breaks that disrupt formatting.
#             - Add capitalization, punctuation, line breaks, paragraphs and other formatting where necessary.
#             - Do NOT change any content or spelling whatsoever.

#             # OUTPUT INSTRUCTIONS

#             - Output the full, properly-formatted text.
#             - Do not output warnings or notes—just the requested sections.

#             # INPUT:

#             INPUT:
#             """},
#             {"role": "user", "content": f"Title: {title}\nDescription: {description}\n\nTranscript to clean:\n{transcript}"}
#         ]
#     )
#     return response.choices[0].message.content


def generate_minutes(client, cleaned_text, title, description):
    MINUTES_PROMPT = """[Your provided prompt here]"""
    
    # Split cleaned text into ~6k token chunks (leaving room for prompt)
    chunks = chunk_text(cleaned_text, max_tokens=6000)
    minutes_sections = []
    
    for i, chunk in enumerate(chunks):
        time.sleep(3)  # Rate limit handling
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": MINUTES_PROMPT},
                    {"role": "user", "content": f"Meeting Title: {title}\nDescription: {description}\nPart {i+1}/{len(chunks)}\n\nTranscript:\n{chunk}"}
                ]
            )
            minutes_sections.append(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating minutes for chunk {i+1}: {str(e)}")
            time.sleep(60)
            continue
    
    # Combine sections intelligently
    combined_minutes = consolidate_minutes_sections(minutes_sections)
    return combined_minutes

def consolidate_minutes_sections(sections):
    """Intelligently merge multiple meeting minutes sections"""
    if len(sections) == 1:
        return sections[0]
        
    # Initialize containers for each section type
    agenda_topics = []
    discussion_topics = []
    recommendations = []
    action_items = []
    
    # Parse and combine sections
    for section in sections:
        parts = section.split('#')
        for part in parts:
            if 'Meeting Agenda Topics' in part:
                agenda_topics.extend(extract_numbered_items(part))
            elif 'Discussion Topics' in part:
                discussion_topics.extend(extract_numbered_items(part))
            elif 'Recommendation' in part:
                recommendations.extend(extract_bullet_points(part))
            elif 'Action Plan' in part:
                action_items.extend(extract_bullet_points(part))

    # Combine into final format
    return f"""# Meeting Agenda Topics
{format_numbered_list(deduplicate_list(agenda_topics))}

# Discussion Topics
{format_numbered_list(deduplicate_list(discussion_topics))}

# Recommendation
{format_bullet_points(deduplicate_list(recommendations))}

# Action Plan
{format_bullet_points(deduplicate_list(action_items))}"""

def extract_numbered_items(text):
    """Extract numbered items from text"""
    import re
    items = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', text, re.DOTALL)
    return [item.strip() for item in items if item.strip()]

def extract_bullet_points(text):
    """Extract bullet points from text"""
    import re
    items = re.findall(r'-\s*(.*?)(?=-|$)', text, re.DOTALL)
    return [item.strip() for item in items if item.strip()]

def deduplicate_list(items):
    """Remove duplicates while preserving order"""
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]

def format_numbered_list(items):
    return '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))

def format_bullet_points(items):
    return '\n'.join(f"- {item}" for item in items)

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
        st.rerun()

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
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")

elif st.session_state.step == 3:
    st.subheader("Step 3: Review Transcription")
    st.text_area("Transcription", st.session_state.transcription, height=300)
    st.download_button("Download Transcription", st.session_state.transcription, "transcription.txt")
    
    if st.button("Clean Transcript"):
        with st.spinner("Cleaning transcript..."):
            st.session_state.cleaned_text = clean_transcript_in_chunks(
                client, 
                st.session_state.transcription,
                st.session_state.title,
                st.session_state.description
            )
        st.session_state.step = 4
        st.rerun()

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
        st.rerun()

elif st.session_state.step == 5:
    st.subheader("Step 5: Meeting Minutes")
    minutes = st.text_area("Meeting Minutes", st.session_state.meeting_minutes, height=400)
    st.session_state.meeting_minutes = minutes
    st.download_button("Download Meeting Minutes", st.session_state.meeting_minutes, "meeting_minutes.txt")
    
    if st.button("Start New Meeting"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.rerun()