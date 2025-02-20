# Personal AI Assistant

## Overview

The **Personal AI Assistant** is a Streamlit application designed to help users interact with URLs and PDF documents by processing, embedding, and querying their content using OpenAI's language model. This application leverages LangChain's document loaders, text splitters, embeddings, and vector stores to facilitate the processing and querying tasks.

### Key Features

- **URL Processing**: Input multiple URLs, load their content, split the text into chunks, and embed the chunks for querying.
- **PDF Processing**: Upload a PDF document, extract its text, split the text into chunks, and embed the chunks for querying.
- **Query Interface**: Ask questions about the processed URLs and PDFs, and get answers generated by the OpenAI model.
- **PDF Summarization**: Summarize the contents of an uploaded PDF document.

## How It Works

### URL Processing

1. **Input URLs**: The user can input multiple URLs through the sidebar.
2. **Load and Process**: The content of the URLs is loaded and split into manageable chunks.
3. **Embedding**: The chunks are embedded using OpenAI's embeddings and stored for later querying.

### PDF Processing

1. **Upload PDF**: The user uploads a PDF document through the sidebar.
2. **Extract and Process**: The text from the PDF is extracted and split into chunks.
3. **Embedding**: The chunks are embedded using OpenAI's embeddings and stored for later querying.

### Query Interface

1. **Select Data Source**: The user selects whether they want to query the processed URLs or PDFs.
2. **Ask Questions**: The user inputs their query, and the system retrieves relevant chunks and generates an answer using the OpenAI model.

### PDF Summarization

1. **Summarize**: The user can choose to summarize the uploaded PDF, and the system generates a concise summary using the OpenAI model.

## Execution Guide

To run the Personal AI Assistant application, follow these steps:

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Required Python packages (specified in `requirements.txt`)

### Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/leonfullxr/PDF-WEB-AI-Assistant.git
    cd personal-ai-assistant
    ```

2. **Install the Required Packages**

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. **Start the Streamlit Application**

    ```sh
    streamlit run main.py
    ```

2. **Interact with the Application**

    - Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).
    - Use the sidebar to input URLs or upload a PDF.
    - Use the query interface to ask questions about the processed data or summarize a PDF.
    
---

Feel free to customize the repository URL and any other details specific to your project setup.
