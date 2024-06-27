import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

pdf_vectors = None  # Initialize pdf_vectors to None for later checks

def process_pdf(uploaded_file):
    global pdf_vectors
    if uploaded_file:
        st.sidebar.write("Loading and processing PDF...")
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=500)
        pdf_docs = text_splitter.split_text(pdf_text)
        if pdf_docs:
            embeddings = OpenAIEmbeddings(openai_api_key='YOUR API KEY HERE')
            pdf_vectors = FAISS.from_texts(pdf_docs, embeddings)
            st.sidebar.write("PDF processed and embedded successfully.")

def get_pdf_vectors():
    global pdf_vectors
    return pdf_vectors
