from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from tqdm import tqdm
import os
import time
import sys
import csv
import requests

# --- Environment ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    sys.exit("❌ OPENAI_API_KEY not set.")
if not QDRANT_CLUSTER_URL or not QDRANT_API_KEY:
    sys.exit("❌ QDRANT_CLUSTER_URL or QDRANT_API_KEY not set.")

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

# --- Define documents to add ---
web_urls = [
    # Wes McKinney - Python for Data Analysis:
    #"https://wesmckinney.com/book/python-basics", DONE
    "https://wesmckinney.com/book/python-builtin",
    #"https://wesmckinney.com/book/numpy-basics", DONE
    "https://wesmckinney.com/book/pandas-basics",
    "https://wesmckinney.com/book/accessing-data",
    "https://wesmckinney.com/book/data-cleaning",
    "https://wesmckinney.com/book/data-wrangling",
    "https://wesmckinney.com/book/plotting-and-visualization",
    "https://wesmckinney.com/book/data-aggregation",
    "https://wesmckinney.com/book/time-series",
    "https://wesmckinney.com/book/modeling",
    "https://wesmckinney.com/book/data-analysis-examples",
    "https://wesmckinney.com/book/advanced-numpy",
    # 3Blue1Brown: Calculus
    "https://www.3blue1brown.com/lessons/essence-of-calculus",
    "https://www.3blue1brown.com/lessons/derivatives",
    "https://www.3blue1brown.com/lessons/derivatives-power-rule",
    "https://www.3blue1brown.com/lessons/derivatives-trig-functions",
    "https://www.3blue1brown.com/lessons/chain-rule-and-product-rule",
    "https://www.3blue1brown.com/lessons/eulers-number",
    "https://www.3blue1brown.com/lessons/implicit-differentiation",
    "https://www.3blue1brown.com/lessons/limits",
    "https://www.3blue1brown.com/lessons/l-hopitals-rule",
    "https://www.3blue1brown.com/lessons/integration",
    "https://www.3blue1brown.com/lessons/area-and-slope",
    "https://www.3blue1brown.com/lessons/higher-order-derivatives",
    "https://www.3blue1brown.com/lessons/taylor-series",
    "https://www.3blue1brown.com/lessons/taylor-series-geometric-view",
    # 3Blue1Brown: Linear Algebra
    #"https://www.3blue1brown.com/lessons/vectors", DONE
    "https://www.3blue1brown.com/lessons/span",
    "https://www.3blue1brown.com/lessons/linear-transformations",
    "https://www.3blue1brown.com/lessons/matrix-multiplication",    
    "https://www.3blue1brown.com/lessons/3d-transformations",
    "https://www.3blue1brown.com/lessons/determinant",
    "https://www.3blue1brown.com/lessons/inverse-matrices",
    "https://www.3blue1brown.com/lessons/nonsquare-matrices",
    "https://www.3blue1brown.com/lessons/dot-products",
    "https://www.3blue1brown.com/lessons/cross-products",
    "https://www.3blue1brown.com/lessons/cross-products-extended",
    "https://www.3blue1brown.com/lessons/cramers-rule",
    "https://www.3blue1brown.com/lessons/change-of-basis",
    "https://www.3blue1brown.com/lessons/eigenvalues",
    "https://www.3blue1brown.com/lessons/quick-eigen",
    "https://www.3blue1brown.com/lessons/abstract-vector-spaces",
    # 3Blue1Brown: Probability
    "https://www.3blue1brown.com/lessons/clt",
    "https://www.3blue1brown.com/lessons/gaussian-integral",
    "https://www.3blue1brown.com/lessons/convolutions2",
    "https://www.3blue1brown.com/lessons/gaussian-convolution",
    "https://www.3blue1brown.com/lessons/bayes-theorem",
    "https://www.3blue1brown.com/lessons/bayes-theorem-quick",
    "https://www.3blue1brown.com/lessons/better-bayes",
    "https://www.3blue1brown.com/lessons/binomial-distributions",
    "https://www.3blue1brown.com/lessons/pdfs",
    # Python Companion to Stat Thinking in 21st Century
    #"https://statsthinking21.github.io/statsthinking21-python/01-IntroductionToPython.html", DONE
    "https://statsthinking21.github.io/statsthinking21-python/02-SummarizingData.html",
    "https://statsthinking21.github.io/statsthinking21-python/03-DataVisualization.html",
    "https://statsthinking21.github.io/statsthinking21-python/04-FittingSimpleModels.html",
    "https://statsthinking21.github.io/statsthinking21-python/05-Probability.html",
    "https://statsthinking21.github.io/statsthinking21-python/06-Sampling.html",
    "https://statsthinking21.github.io/statsthinking21-python/07-ResamplingAndSimulation.html",
    "https://statsthinking21.github.io/statsthinking21-python/08-HypothesisTesting.html",
    "https://statsthinking21.github.io/statsthinking21-python/09-StatisticalPower.html",
    "https://statsthinking21.github.io/statsthinking21-python/10-BayesianStatistics.html",
    "https://statsthinking21.github.io/statsthinking21-python/11-ModelingCategoricalRelationships.html",
    "https://statsthinking21.github.io/statsthinking21-python/13-GeneralLinearModel.html"

]

pdf_paths = [
 #   "course_materials/brucePracStatsForDS.pdf" ALL DONE
]

# --- Custom Web Loader with User-Agent ---
class CustomWebLoader(WebBaseLoader):
    def __init__(self, url):
        super().__init__(url)
        self.requests_kwargs = {
            "headers": {
                "User-Agent": "INFO511-AI-TA/1.0"
            }
        }

# --- Ingest new content only ---
def ingest(batch_size=10):
    all_docs = []
    sources_to_add = []

    for url in tqdm(web_urls, desc="🌐 Checking Web URLs"):
        if url in logged_sources:
            print(f"✅ Already embedded: {url}")
            continue
        try:
            docs = CustomWebLoader(url).load()
            for doc in docs:
                doc.metadata["source"] = url
            all_docs.extend(docs)
            sources_to_add.append(url)
        except Exception as e:
            print(f"❌ Failed to load {url}: {e}")

    for path in tqdm(pdf_paths, desc="📄 Checking Local PDFs"):
        if path in logged_sources:
            print(f"✅ Already embedded: {path}")
            continue
        try:
            docs = PyPDFLoader(path).load()
            for doc in docs:
                doc.metadata["source"] = path
            all_docs.extend(docs)
            sources_to_add.append(path)
        except Exception as e:
            print(f"❌ Failed to load PDF {path}: {e}")

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

    deduplicate_csv("embedded_sources.csv")
    print("✅ Embedding complete.")

# --- Deduplicate CSV ---
def deduplicate_csv(path="embedded_sources.csv"):
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = set(tuple(row) for row in csv.reader(f))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sorted(rows))

if __name__ == "__main__":
    ingest()
