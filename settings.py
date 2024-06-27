import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Personal AI Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("<h1 style='text-align: center; color: white;'>Personal AI Assistant</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center; color: white;'>Assistant Console</h3>", unsafe_allow_html=True)

