import streamlit as st
from langchain.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import pandas as pd
import re
import requests
import json
from datetime import datetime
from fpdf import FPDF
import tempfile

# ✅ MUST come before any other st.something() call
st.set_page_config(page_title="AI Tutor for INFO 511", page_icon="🧑‍🏫")

# --- Config ---
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_SHEET_URL = os.getenv("GITHUB_SHEET_URL")
ADMIN_USERS = {"gchism94"}  # lowercase!

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY")
    st.stop()

# --- Logging ---
logging.basicConfig(filename="chat_log.txt", level=logging.INFO, format="%(asctime)s %(message)s")
ethics_logger = logging.getLogger("ethics")
ethics_handler = logging.FileHandler("ethics_log.txt")
ethics_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
ethics_logger.setLevel(logging.INFO)
ethics_logger.addHandler(ethics_handler)

def send_slack_alert(user_identifier, flagged_text):
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if not slack_url:
        return
    message = f"🚨 *Ethics Flag Triggered*\nUser: `{user_identifier}`\nQuery: ```{flagged_text}```"
    try:
        requests.post(slack_url, json={"text": message})
    except Exception as e:
        logging.error(f"Failed to send Slack alert: {e}")

def ethics_warning(text, user_identifier="unknown", display_name=None):
    text = text.lower()
    red_flags = [
        r"can you (solve|answer|do) .* (assignment|exam|test|quiz)",
        r"what is the answer to .*",
        r"give me the solution to .*",
        r"complete this for me",
        r"write my .* homework",
        r"plagiar[ize|ise]",
        r"copy this",
        r"cheat(ing)?",
        r"how to bypass",
        r"give me the code for .* assignment",
        r"do my coursework",
        r"impersonate.*student",
        r"submit this as me",
        r"fool (chatgpt|ai) detector",
        r"make this undetectable",
    ]
    for pattern in red_flags:
        if re.search(pattern, text):
            ethics_logger.info(f"⚠️ ETHICS WARNING from {user_identifier} (displayed as '{display_name}'): {text}")
            send_slack_alert(user_identifier, text)
            return True
    return False

def format_quiz_with_reveal(quiz_text):
    import re
    questions = quiz_text.strip().split("Q")
    formatted_blocks = []
    for q in questions:
        if not q.strip():
            continue
        q = "Q" + q.strip()
        match = re.search(r"Q\d+\..*?(?=A\.)", q, re.DOTALL)
        question_text = match.group(0).strip() if match else q.strip()
        options = re.findall(r"([A-D])\.\s*(.*?)\s*(?=(?:[A-D]\.|Answer:|$))", q)
        answer_match = re.search(r"Answer:\s*([A-D])", q)
        correct = answer_match.group(1) if answer_match else None

        md = f"**{question_text}**\n\n"
        for letter, option in options:
            if letter == correct:
                md += f"- **{letter}. {option}**\n"
            else:
                md += f"- {letter}. {option}\n"
        md += f"\n<details><summary>✅ Reveal Answer</summary>\n\n**Correct Answer: {correct}**\n</details>\n"
        formatted_blocks.append(md)

    return "\n\n---\n\n".join(formatted_blocks)

def generate_quiz_pdf(user_identifier, quiz_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.set_title("AI TA - Quiz History")
    pdf.cell(200, 10, txt="Quiz History Export", ln=True, align="C")
    pdf.ln(10)

    for entry in quiz_history:
        pdf.multi_cell(0, 10, f"Question: {entry['question']}")
        pdf.ln(1)
        pdf.multi_cell(0, 10, f"AI Response: {entry['ai_response']}")
        pdf.ln(1)
        cleaned_quiz = format_quiz_with_reveal(entry["quiz"]).replace("✅", "Answer:")
        cleaned_quiz = re.sub(r"<.*?>", "", cleaned_quiz)  # Remove HTML tags
        pdf.multi_cell(0, 10, f"Quiz:\n{cleaned_quiz}")
        pdf.ln(1)
        pdf.cell(0, 10, f"Confidence: {entry.get('confidence', '?')} / 5", ln=True)
        pdf.ln(10)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name

# --- GitHub Auth ---
@st.cache_data(ttl=300)
def load_allowed_github_users():
    try:
        df = pd.read_csv(GITHUB_SHEET_URL)
        df["github_username"] = df["github_username"].str.lower()
        return set(df["github_username"].dropna().unique())
    except Exception as e:
        st.error(f"Failed to load GitHub user list: {e}")
        return set()

allowed_githubs = load_allowed_github_users()

if "user_identifier" not in st.session_state:
    github_username = st.text_input("Enter your GitHub username to begin:").strip().lower()
    if github_username:
        if github_username in allowed_githubs:
            st.session_state["user_identifier"] = github_username
            st.rerun()
        else:
            st.warning("Please enter a valid GitHub username.")
    st.stop()

user_identifier = st.session_state["user_identifier"]

# --- Ensure display_name is always set
if "display_name" not in st.session_state:
    st.session_state["display_name"] = user_identifier

display_name = st.session_state["display_name"]

is_admin = user_identifier in ADMIN_USERS


# --- Display name preference ---
# --- Display name preference ---
if "display_name" not in st.session_state:
    with st.expander("🔒 Optional: Customize Display Name", expanded=False):
        name_input = st.text_input("Preferred name (optional)", "")
        consent = st.checkbox("Remember this display name on this device?")
        if name_input.strip():
            st.session_state["display_name"] = name_input.strip()
            if consent:
                st.cache_data(ttl=86400)(lambda x: x)(name_input.strip())  # pseudo-cache locally
        else:
            st.session_state["display_name"] = user_identifier

# --- Determine and show display name
display_name = st.session_state["display_name"]

if display_name != user_identifier:
    st.markdown(f"📝 Signed in as **{display_name}** &nbsp;&nbsp;&nbsp;<span style='font-size: 0.9em;'>(GitHub: `{user_identifier}`)</span>", unsafe_allow_html=True)
else:
    st.markdown(f"📝 Signed in as `{user_identifier}`")


# --- Welcome + display pill ---
display_name = st.session_state["display_name"]
st.markdown(f"📝 Signed in as **{display_name}** (`{user_identifier}`)")
st.success(f"Welcome, {display_name}! 👋")

# --- Onboarding ---
if "onboarded" not in st.session_state:
    st.header("🚀 Getting Started with Your AI Tutor")
    st.markdown("""
    Welcome to the **AI Powered Tutor** for INFO 511.

    This tool is designed to help you:
    - 📘 Understand data science concepts
    - 💡 Generate pseudocode
    - 🔍 Explore documentation
    - 🧠 Quiz yourself adaptively

    ---
    ### ⚠️ Ethical Use Reminder
    - Do **not** ask the AI to complete your assignments for you.
    - Do **not** submit AI-generated responses as your own work.
    - **We log interactions** for academic integrity and learning improvements.

    ---
    """)

    if st.button("✅ I Understand and Agree — Let’s Go!"):
        st.session_state.onboarded = True
        st.rerun()
    else:
        st.stop()

# --- Track usage hour ---
now = datetime.now()
hour = now.hour

if hour >= 9 and hour <= 5:
    logging.info(f"🌙 LATE-NIGHT USAGE: {user_identifier} ({display_name}) used the app at {now.isoformat()}")


# --- Set up LangChain chain ---
@st.cache_resource
def load_retriever():
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Qdrant.from_existing_collection(
        collection_name="ai_ta_sources",
        url=QDRANT_CLUSTER_URL,
        api_key=QDRANT_API_KEY,
        embedding=embedding,
        path=None
    )
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})


retriever = load_retriever()
llm = Ollama(model="mistral", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- UI ---
st.title("🧑‍🏫 INFO 511 AI Teaching Assistant")
st.markdown("I can explain data science concepts, write pseudocode, and cite documentation.")

intent = st.selectbox("What do you want help with?", ["Explain concept", "Write pseudocode", "Compare methods", "Debug code", "Other"])
source_filter = st.multiselect("📚 Limit search to these sources (optional):", 
    ["wesmckinney.com", "3blue1brown.com", "statsthinking21.github.io", "course_materials"], default=[])

if is_admin:
    with st.expander("🧠 System Prompt (Admin Only)"):
        system_prompt = st.text_area(
            "Edit System Instructions:",
            value="You are an AI TA for INFO 511. Only give helpful explanations and pseudocode. Never provide full working code. Cite docs where useful.",
            height=150
        )
else:
    system_prompt = "You are an AI TA for INFO 511. Only give helpful explanations and pseudocode. Never provide full working code. Cite docs where useful."
    st.caption("🧠 System instructions are fixed for consistency. If you have feedback, let the instructor know.")


# --- Upload Assignments ---
if "assignments" not in st.session_state:
    st.session_state.assignments = []
    st.session_state.violations = {}

if is_admin:
    st.markdown("### 🔒 Admin: Upload Assignment File(s)")
    uploaded = st.file_uploader("Upload assignment file(s)", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            try:
                text = file.read().decode("utf-8")
                st.session_state.assignments.append((file.name, text))
                st.success(f"✅ {file.name} uploaded.")
            except Exception as e:
                st.warning(f"❌ Failed to read {file.name}: {e}")
else:
    st.caption("📎 Assignment uploads are restricted to instructors.")

# --- Chat Input and QA ---
user_input = st.chat_input("Ask a question related to INFO 511...")
if "history" not in st.session_state:
    st.session_state.history = []

if user_input:
    if ethics_warning(user_input, user_identifier, display_name):
        st.info("⚠️ This question was flagged for ethical review. No response will be generated.")
        st.stop()
    assignment_texts = [text for _, text in st.session_state.assignments]
    similarity_score = 0
    if assignment_texts:
        vectorizer = TfidfVectorizer().fit_transform(assignment_texts + [user_input])
        sim_matrix = cosine_similarity(vectorizer)
        similarity_score = max(sim_matrix[-1][:-1])

        violation_key = user_identifier.lower()
        if similarity_score > 0.75:
            st.session_state.violations[violation_key] = st.session_state.violations.get(violation_key, 0) + 1
            ethics_logger.info(f"⚠️ SIMILARITY FLAG from {user_identifier}: {user_input}")
            if st.session_state.violations[violation_key] >= 3:
                st.error("🚫 You have reached the limit for assignment-related queries.")
                st.stop()
            else:
                st.warning(f"⚠️ This question is very similar to an assignment. {3 - st.session_state.violations[violation_key]} chances left.")

    filter_clause = f"\n\nOnly use documents from: {', '.join(source_filter)}" if source_filter else ""
    query = f"{system_prompt}\n\nIntent: {intent}{filter_clause}\n\nStudent Question: {user_input}"
    result = qa_chain({"query": query})

    st.session_state.history.append({
        "user": user_input,
        "ai": result["result"],
        "sources": result["source_documents"]
    })
    logging.info(f"USER: {user_identifier}\nQUESTION: {user_input}\nANSWER: {result['result']}\n")

# --- Display history and quiz ---
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = {}

for turn in st.session_state.history[::-1]:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("ai"):
        st.markdown(turn["ai"])
        with st.expander("📄 Sources"):
            for i, doc in enumerate(turn["sources"]):
                src = doc.metadata.get("source", "unknown")
                st.markdown(f"[{i+1}]({src})", unsafe_allow_html=True)

    if st.button("📝 Generate Quiz", key=f"quiz_{turn['user']}"):
        with st.spinner("Creating an adaptive quiz..."):
            quiz_prompt = f"""
            You are an AI tutor for a data science course. Based on the student's question and your explanation, generate a short quiz with 3 multiple-choice questions (A–D). Indicate the correct answer.

            Student Question:
            {turn['user']}

            Your Explanation:
            {turn['ai']}

            Format:
            Q1. ...
            A. ...
            B. ...
            C. ...
            D. ...
            Answer: B
            """
            quiz_response = llm.invoke(quiz_prompt)
            st.markdown("### 📚 Adaptive Quiz")
            st.markdown(format_quiz_with_reveal(quiz_response), unsafe_allow_html=True)

            # 🌟 New: Confidence input
            confidence = st.slider("How confident are you about this material?", 1, 5, 3)

            # ✅ Save quiz + confidence
            st.session_state.quiz_history.setdefault(user_identifier, []).append({
                "question": turn["user"],
                "ai_response": turn["ai"],
                "quiz": quiz_response,
                "confidence": confidence
            })

if st.button("📥 Download My Quiz History as JSON"):
    raw_quiz_data = st.session_state.quiz_history.get(user_identifier, [])
    cleaned = [
        {
            "question": entry["question"],
            "ai_response": entry["ai_response"],
            "quiz_raw": entry["quiz"],
            "quiz_formatted": format_quiz_with_reveal(entry["quiz"]),
            "confidence": entry.get("confidence")
        }
        for entry in raw_quiz_data
    ]
    st.download_button(
        "Download JSON",
        json.dumps(cleaned, indent=2),
        file_name="quiz_history_cleaned.json"
    )

if st.button("📄 Download My Quiz History as PDF"):
    raw_quiz_data = st.session_state.quiz_history.get(user_identifier, [])
    if raw_quiz_data:
        pdf_path = generate_quiz_pdf(user_identifier, raw_quiz_data)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="quiz_history_export.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("You have no quiz history to download.")

