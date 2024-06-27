import streamlit as st
import pickle
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

url_docs_file_path = "url_docs.pkl"

def process_urls():
    num_links = st.sidebar.slider("How many links do you want to input?", min_value=1, max_value=5, value=1)
    urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url{i}") for i in range(num_links)]

    if urls:
        st.sidebar.write("Loading and processing URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=1000)
        url_docs = text_splitter.split_documents(data)
        if url_docs:
            with open(url_docs_file_path, "wb") as f:
                pickle.dump(url_docs, f)
            st.sidebar.write("URLs processed and embedded successfully.")

def load_url_embeddings():
    if os.path.exists(url_docs_file_path):
        with open(url_docs_file_path, "rb") as f:
            url_docs = pickle.load(f)
        return url_docs
    return None

