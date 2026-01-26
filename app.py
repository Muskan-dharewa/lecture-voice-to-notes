import streamlit as st
import whisper
import os
import tempfile
from openai import OpenAI

st.set_page_config(page_title="Lecture Voice-to-Notes Generator", layout="wide")

# --- OpenAI client ---
client = OpenAI()

# --- Load Whisper model ---
@st.cache_resource
def load_whisper():
    # Use "small" for testing (faster). Switch to "medium" later.
    return whisper.load_model("small")

whisper_model = load_whisper()

# --- Helper functions ---
def chunk_text(text, chunk_size=1800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_chunk(chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize this lecture section in 3–4 clear lines:\n{chunk}"
        }],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_study_material(summary_text):
    prompt = f"""
You are an expert teacher.

From the lecture summary below:

- Focus on concepts, causes, effects, and meanings
- Ignore objects, places, and trivial nouns
- Use simple student-friendly language

Tasks:
1. Write a refined summary (7–8 lines).
2. Extract 12–15 meaningful academic key topics.
3. Generate 15 conceptual quiz questions.
4. Create 15 flashcards in Question–Answer format.

Lecture summary:
{summary_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# --- UI ---
st.title("🎓 Lecture Voice-to-Notes Generator")
st.write("Upload a lecture audio file and generate transcript, notes, quiz and flashcards.")

audio_file = st.file_uploader("Upload Lecture Audio (.mp3 or .wav)", type=["mp3", "wav"])
generate_btn = st.button("🚀 Generate Study Material")

if audio_file is not None and generate_btn:

    # Save uploaded audio safely using temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.success("Audio uploaded successfully ✅")

    progress = st.progress(0)

    # --- Transcription ---
    with st.spinner("🔊 Transcribing full lecture... (may take time)"):
        result = whisper_model.transcribe(audio_path, language="en", fp16=False)
        transcript = result["text"]
    progress.progress(30)

    st.subheader("📜 Full Lecture Transcript")
    st.text_area("Transcript", transcript, height=250)

    # --- Chunking ---
    chunks = chunk_text(transcript)
    st.info(f"Lecture split into {len(chunks)} parts for processing.")
    progress.progress(40)

    # --- Chunk summaries ---
    chunk_summaries = []
    with st.spinner("🧠 Summarizing lecture sections..."):
        for i, chunk in enumerate(chunks, 1):
            chunk_summaries.append(summarize_chunk(chunk))
    combined_summary = " ".join(chunk_summaries)

    # --- Final study material ---
    with st.spinner("📘 Generating final notes, quiz and flashcards..."):
        study_material = generate_study_material(combined_summary)
    progress.progress(100)

    st.subheader("📘 Generated Study Material")
    st.markdown(study_material)

    # Cleanup temp file
    try:
        os.remove(audio_path)
    except:
        pass
