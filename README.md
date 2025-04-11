# 🧠 AI Tutor (DEVELOPMENTAL soon to be JupyterQuest+)

An open-source AI-powered tutor built to support student learning through adaptive quizzes, coding help, and ethical guidance.

> ⚠️ **Note**: This is an active developmental version of the AI Teaching Assistant.  
> The system is not yet production-ready and may undergo significant changes. 
>The current version is in the process of being consolidated into [jupyterquest](https://github.com/Gchism94/jupyterquest).  
> Contributions, feedback, and testing are welcome!

**JupyterQuest+** is designed to be embedded in data science courses as a virtual TA. It uses large language models (LLMs) to answer coding questions, generate personalized quizzes, and flag ethically sensitive responses—all while helping instructors keep tabs on student engagement and integrity.

---

## ✨ Features

- 📚 **Adaptive Quiz Engine** – Dynamically generates questions based on student performance
- 🤖 **AI-Powered Tutor** – Answers student questions about Python, data science, and statistics
- 🧠 **Ethics-Aware AI** – Detects ethically sensitive prompts and applies nudges or restrictions
- 📊 **Quiz Tracking** – Stores quiz scores per student for personalized recommendations
- 🔔 **Professor Alerts** – Optionally sends Slack alerts when ethical triggers are detected
- 🔍 **Similarity Detection** – Flags assignment uploads for plagiarism using vector search
- 🧩 **RAG-Enhanced Help** – Retrieval-Augmented Generation from course materials

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/ai-tutor.git
cd ai-tutor
```

### 2. Install dependencies
Use `venv`, `conda`, or `pipenv`, then:

```bash
pip install -r requirements.txt
```

### 3. Set your environment variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-key
QDRANT_API_KEY=your-qdrant-key
SLACK_WEBHOOK_URL=optional-slack-url
```

### 4. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
ai-teaching-assistant/
│
├── .streamlit/                  # Streamlit config (e.g., secrets.toml)
├── assignments/                 # Uploaded assignments (you might want to .gitignore this)
├── chroma_db/                  # Vector DB (likely should be ignored or mounted)
├── course_materials/           # Source documents for retrieval
│
├── src/                         # Source code
│   ├── app.py                   # Entry point (rename best version to app.py)
│   ├── ingest.py                # Document ingestion logic
│   ├── add_documents.py         # (could be merged with ingest.py if similar)
│   ├── utils/                   # Helper modules (if needed)
│   ├── ethics/                 # Ethics-related logging & logic
│   │   └── ethics_log.txt       # (move this or convert to CSV/JSON/log format)
│   └── logs/
│       └── chat_log.txt
│
├── data/                        # Any embedded or uploaded data (to be ignored in .gitignore)
│   └── embedded_sources.csv
│
├── docker/                      # Docker-related files
│   ├── Dockerfile               # Default Dockerfile
│   ├── Dockerfile.prod
│   ├── Dockerfile-no-ollama
│   ├── docker-compose.yml
│   └── docker-compose.prod.yaml
│
├── .gitignore
├── .dockerignore
├── .env                         # .gitignored
├── README.md
├── requirements.txt
└── LICENSE                      # Add one if open-source
```

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – For the user interface
- **OpenAI / Ollama** – LLM backend (supports both OpenAI and local models)
- **Qdrant** – Vector database for document retrieval & similarity detection
- **Slack Webhooks** – For instructor notifications (optional)
- **GitHub Actions** – For autograding and daily imports (planned)

---

## 📈 Planned Enhancements

- 🔁 Integration with GitHub Classroom and LMS systems
- 🧪 Automatic grading of quizzes and coding assignments
- 🧭 Custom course personas for contextualized tutoring
- 🗂 Dashboard for instructors to monitor student progress

---

## 🧑‍🏫 Built By

**Greg T. Chism, Ph.D.**  
Assistant Professor of Practice  
Coordinator of Data Science Student Engagement  
University of Arizona College of Information Science  
[Website](https://gregtchism.com) | [GitHub](https://github.com/Gchism94)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.