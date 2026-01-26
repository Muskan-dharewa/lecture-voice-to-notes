import streamlit as st
import whisper
import os
import tempfile
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Lecture Voice-to-Notes Generator", layout="wide")

st.title("🎓 Lecture Voice-to-Notes Generator")
st.caption("Upload lecture audio → get transcript, notes, topics, quiz & flashcards")

# ---------------- OPENAI CLIENT ----------------
client = OpenAI()  # Uses OPENAI_API_KEY from environment

# ---------------- LOAD WHISPER ----------------
@st.cache_resource
def load_whisper(model_size: str):
    return whisper.load_model(model_size)

# Sidebar UI
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Whisper Model (Speed vs Accuracy)",
    ["small (fast)", "medium (accurate)"],
    index=0
)

whisper_model_name = "small" if "small" in model_choice else "medium"
whisper_model = load_whisper(whisper_model_name)

chunk_size = st.sidebar.slider("Chunk Size (text split)", 1500, 3500, 2500, step=500)
num_flashcards = st.sidebar.slider("Flashcards", 5, 25, 15)
num_questions = st.sidebar.slider("Quiz Questions", 5, 25, 15)

audio_file = st.sidebar.file_uploader("📤 Upload Audio (.mp3/.wav)", type=["mp3", "wav"])
generate_btn = st.sidebar.button("🚀 Generate")

# ---------------- HELPER FUNCTIONS ----------------
def chunk_text(text, size=2500):
    return [text[i:i + size] for i in range(0, len(text), size)]

def summarize_chunk(chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize this lecture part in 3–4 clear student-friendly lines:\n{chunk}"
        }],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_final_study_material(combined_summary, q_count, f_count):
    prompt = f"""
You are an expert teacher preparing exam-oriented study material.

From the lecture summary below:

1) Write a refined final summary (7–8 lines).
2) Extract 10–12 meaningful academic key topics (not objects).
3) Generate {q_count} conceptual quiz questions (focus on concepts, causes, effects).
4) Create {f_count} flashcards in Question–Answer format with complete answers.

Rules:
- Ignore trivial nouns like desk/board/people.
- Use simple language.
- Make content useful for revision.

Lecture summary:
{combined_summary}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------------- MAIN LOGIC ----------------
if audio_file is not None and generate_btn:

    # Save audio safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.success("✅ Audio uploaded successfully!")

    progress = st.progress(0)

    # ---- TRANSCRIPTION ----
    with st.spinner("🔊 Transcribing full lecture..."):
        result = whisper_model.transcribe(audio_path, language="en", fp16=False)
        transcript = result["text"]
    progress.progress(30)

    # Show transcript in tab
    tab1, tab2, tab3 = st.tabs(["📜 Transcript", "📘 Notes", "❓ Quiz & Flashcards"])

    with tab1:
        st.subheader("Full Lecture Transcript")
        st.text_area("Transcript", transcript, height=300)

    # ---- CHUNKING ----
    chunks = chunk_text(transcript, size=chunk_size)
    st.info(f"Lecture split into {len(chunks)} parts for processing.")
    progress.progress(40)

    # ---- CHUNK SUMMARIES ----
    with st.spinner("🧠 Summarizing lecture parts..."):
        summaries = []
        for i, chunk in enumerate(chunks):
            summaries.append(summarize_chunk(chunk))
            progress.progress(40 + int(((i + 1) / len(chunks)) * 40))

    combined_summary = " ".join(summaries)

    # ---- FINAL STUDY MATERIAL ----
    with st.spinner("📘 Generating final study material..."):
        final_output = generate_final_study_material(
            combined_summary,
            q_count=num_questions,
            f_count=num_flashcards
        )
    progress.progress(100)

    with tab2:
        st.subheader("📘 Generated Notes / Summary / Topics")
        st.markdown(final_output)

    with tab3:
        st.subheader("❓ Quiz + Flashcards")
        st.markdown(final_output)

    # Cleanup temp audio file
    try:
        os.remove(audio_path)
    except:
        pass

elif audio_file is not None and not generate_btn:
    st.info("⬅️ Click **Generate** in the sidebar to start processing.")

else:
    st.info("⬅️ Upload an audio file from the sidebar to begin.")
