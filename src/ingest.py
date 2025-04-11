from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from tqdm import tqdm
import os
import time
import sys
import csv
import zipfile
import glob
import tempfile

# --- Environment ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    sys.exit("❌ OPENAI_API_KEY not set.")
    
if not QDRANT_CLUSTER_URL or not QDRANT_API_KEY:
    sys.exit("❌ QDRANT_CLUSTER_URL or QDRANT_API_KEY not set.")

# --- Ethics filter ---
def is_ethically_concerning(text):
    flag_keywords = ["cheat", "plagiarize", "answer the test", "do my assignment"]
    return any(kw in text.lower() for kw in flag_keywords)

# --- Initialize Qdrant store ---
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(
    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY,
)
qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    embedding=embedding,
    collection_name="ai_ta_sources"
)

# --- Load processed sources ---
def load_logged_sources():
    if not os.path.exists("embedded_sources.csv"):
        return set()
    with open("embedded_sources.csv", newline="") as f:
        return set(row[0] for row in csv.reader(f) if row)

logged_sources = load_logged_sources()

# --- Sources to ingest ---
web_urls = []
pdf_paths = []
assignment_web_urls = [
    # Add assignment web pages here:
    "https://datasciaz.netlify.app/ae/ae-00-unvotes.html",
    "https://datasciaz.netlify.app/assignments/project-01.html",
    "https://datasciaz.netlify.app/ae/ae-01-meet-the-penguins.html",
    "https://datasciaz.netlify.app/ae/ae-02-diwali-sales.html",
    "https://datasciaz.netlify.app/ae/ae-03-tucson-housing.html",
    "https://datasciaz.netlify.app/ae/ae-04-flights-preprocessing.html",
    "https://datasciaz.netlify.app/ae/ae-05-majors-wrangling.html",
    "https://datasciaz.netlify.app/ae/ae-06-wildcat-scrape.html",
    "https://datasciaz.netlify.app/ae/ae-07-probability.html",
    "https://datasciaz.netlify.app/ae/ae-08-conditional-probability.html",
    "https://datasciaz.netlify.app/ae/ae-09-hypothesis-testing.html",
    "https://datasciaz.netlify.app/ae/ae-10-modeling-fish.html",
    "https://datasciaz.netlify.app/ae/ae-11-modeling-loans.html",
    "https://datasciaz.netlify.app/ae/ae-12-spam-filter.html",
    "https://datasciaz.netlify.app/ae/ae-13-candy-ranking.html",
    "https://datasciaz.netlify.app/ae/ae-14-derivation.html",
    "https://datasciaz.netlify.app/ae/ae-15-integration.html",
    "https://datasciaz.netlify.app/ae/ae-16-linear-algebra.html",
    "https://datasciaz.netlify.app/ae/ae-17-pca.html",
    "https://datasciaz.netlify.app/summative/final.html",
    "https://datasciaz.netlify.app/summative/midterm.html"
]

zip_paths = glob.glob("assignments/*.zip")

# --- Helper: load documents from extracted zip files ---
def load_text_files(directory, source_tag):
    docs = []
    for ext in (".py", ".ipynb", ".md", ".html"):
        for path in glob.glob(f"{directory}/**/*{ext}", recursive=True):
            try:
                loader = (
                    NotebookLoader(path) if path.endswith(".ipynb")
                    else TextLoader(path, autodetect_encoding=True)
                )
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = f"{source_tag}::{os.path.basename(path)}"
                    if is_ethically_concerning(doc.page_content):
                        print(f"⚠️ Ethics flag in file {path}.")
                docs.extend(loaded)
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
    return docs

# --- Ingest process ---
def ingest(batch_size=10):
    all_docs = []
    sources_to_add = []

    for path in tqdm(zip_paths, desc="📦 Unpacking ZIP Assignments"):
        if path in logged_sources:
            print(f"✅ Already embedded: {path}")
            continue
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                docs = load_text_files(tmpdir, source_tag=path)
                all_docs.extend(docs)
                sources_to_add.append(path)
        except Exception as e:
            print(f"❌ Failed to unpack {path}: {e}")

    for url in tqdm(assignment_web_urls, desc="🌐 Loading Assignment Web Pages"):
        if url in logged_sources:
            print(f"✅ Already embedded: {url}")
            continue
        try:
            docs = WebBaseLoader(url).load()
            for doc in docs:
                doc.metadata["source"] = f"assignment::{url}"
                if is_ethically_concerning(doc.page_content):
                    print(f"⚠️ Ethics flag in {url}")
            all_docs.extend(docs)
            sources_to_add.append(url)
        except Exception as e:
            print(f"❌ Failed to load {url}: {e}")

            
    if not all_docs:
        print("✅ No new documents to embed.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    for i in tqdm(range(0, len(chunks), batch_size), desc="🔁 Embedding batches"):
        batch = chunks[i:i + batch_size]
        qdrant_store.add_documents(batch)
        time.sleep(2)

    with open("embedded_sources.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for source in sources_to_add:
            writer.writerow([source])

    print("✅ Embedding complete.")

if __name__ == "__main__":
    ingest()
