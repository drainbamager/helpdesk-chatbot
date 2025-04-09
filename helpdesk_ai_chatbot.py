import os
import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.readers import SimpleWebPageReader  # Updated import path for SimpleWebPageReader
from llama_index import Document  # Adjusted import path for Document
from bs4 import BeautifulSoup
import requests

# === Configuration ===
# Load your OpenAI key securely (don't hardcode in production)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set up LLM
llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

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
        text = soup.get_text()
        documents.append(Document(text=text, metadata={"source": url}))
    return documents

@st.cache_resource(show_spinner=True)
def load_index():
    st.write("Loading and indexing documents...")
    helpjuice_docs = fetch_helpjuice_articles()
    event_docs = scrape_event_site()
    all_docs = helpjuice_docs + event_docs
    index = VectorStoreIndex.from_documents(all_docs, service_context=service_context)
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
        st.markdown(f"- {node.node.metadata.get('source', 'Unknown')}")

st.markdown("---")
st.markdown("Built with 🧠 using LlamaIndex + OpenAI + Streamlit")