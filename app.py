import streamlit as st
import whisper
import os
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Lecture Voice-to-Notes Generator",
    layout="wide"
)

client = OpenAI()  # API key read from environment

# ---------------- LOAD WHISPER ----------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ---------------- LLM FUNCTION ----------------
def generate_study_material(transcript):
    prompt = f"""
You are an academic study assistant.

From the lecture text below, do the following:

1. Write a clear student-friendly summary (7-8 lines).
2. Extract 15–20 meaningful academic key topics (not objects).
3. Generate 25 conceptual quiz questions.
4. Create 20-25 flashcards (Q&A format).

Lecture text:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.title("🎓 Lecture Voice-to-Notes Generator")
st.write("Upload a lecture audio file and get **clean notes, topics, quizzes, and flashcards**.")

audio_file = st.file_uploader(
    "Upload Lecture Audio (.mp3 or .wav)",
    type=["mp3", "wav"]
)

if audio_file is not None:

    # Save audio
    audio_path = os.path.join(os.getcwd(), "temp_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

    st.success("Audio uploaded successfully!")

    # -------- TRANSCRIPTION --------
    with st.spinner("Transcribing lecture audio..."):
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

    st.subheader("📜 Raw Transcript")
    st.text_area("Transcript", transcript, height=220)

    # -------- GENERATIVE AI --------
    with st.spinner("Generating study material using AI..."):
        study_material = generate_study_material(transcript)

    st.subheader("📘 Generated Study Material")
    st.markdown(study_material)
