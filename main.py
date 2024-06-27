import streamlit as st
from settings import setup_page
from url_processing import process_urls, load_url_embeddings
from pdf_processing import process_pdf, load_pdf_embeddings
from query_interface import handle_query

# Streamlit setup
setup_page()

# Sidebar sections
st.sidebar.subheader("URL Input")
if st.sidebar.button("Process URLs"):
    process_urls()

st.sidebar.subheader("PDF Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=['pdf'])
if st.sidebar.button("Process PDF"):
    process_pdf(uploaded_file)

# Query interface
handle_query(uploaded_file)

