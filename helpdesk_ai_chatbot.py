import os
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.readers.web.simple_web.base import SimpleWebPageReader
from llama_index.core.schema import Document
from bs4 import BeautifulSoup
import requests

# === Configuration ===
# Load your OpenAI key securely (don't hardcode in production)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set up LLM
llm = OpenAI(model="gpt-4")
Settings.llm = llm

# === Load Documents ===
def fetch_helpjuice_articles():
    urls = [
        "https://globalencounters.helpjuice.com/en_US/hospitality-faq"  # Replace with real URLs or use API
    ]
    reader = SimpleWebPageReader()
    docs = reader.load_data(urls)
    return docs

def scrape_event_site():
    urls = [
        "https://the.ismaili/globalencounters/"
    ]
    documents = []
    for url in urls:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Remove unnecessary tags like header, footer, nav, scripts
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()

        # Extract visible text and clean whitespace
        text = ' '.join(soup.stripped_strings)

        documents.append(Document(text=text, metadata={"source": url, "doc_id": url}))
    return documents

@st.cache_resource(show_spinner=True)
def load_index():
    st.write("Loading and indexing documents...")
    helpjuice_docs = fetch_helpjuice_articles()
    event_docs = scrape_event_site()
    all_docs = helpjuice_docs + event_docs
    index = VectorStoreIndex.from_documents(all_docs)
    return index

# === Streamlit UI ===
st.set_page_config(page_title="HelpDesk AI Agent", page_icon="🤖")
st.title("🤖 HelpDesk AI Assistant")

query = st.text_input("Ask a question about the event:")

if query:
    index = load_index()
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    st.markdown("### 💬 Answer:")
    st.write(response.response)
    st.markdown("---")
    st.markdown("### 📚 Sources:")
    for node in response.source_nodes:
        source = node.node.metadata.get('source') or node.node.metadata.get('doc_id') or 'Unknown'
        st.markdown(f"- {source}")

st.markdown("---")
st.markdown("Built with 🧠 using LlamaIndex + OpenAI + Streamlit")