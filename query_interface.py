import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from url_processing import get_url_embeddings
from pdf_processing import get_pdf_vectors

llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key='YOUR API KEY HERE')

def handle_query():
    st.subheader("Query Interface")
    data_source = st.selectbox("What do you want to inquire about?", ["URL", "PDF"])

    if data_source == "URL":
        query_url = st.text_input('Ask your question about URLs:')
        if query_url:
            url_vectorindex_openai = get_url_embeddings()
            if url_vectorindex_openai:
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=url_vectorindex_openai.as_retriever())
                result = chain({"question": query_url}, return_only_outputs=True)
                st.header("Answer based on URLs:")
                st.subheader(result['answer'])

    elif data_source == "PDF":
        query_pdf = st.text_input('Ask your question about PDFs:')
        pdf_vectors = get_pdf_vectors()
        if query_pdf and pdf_vectors:
            docs = pdf_vectors.similarity_search(query_pdf)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query_pdf)
            st.write(response)

        if st.button("Summarize PDF") and pdf_vectors:
            st.write("Summarizing PDF...")

            def summarize_pdfs_from_folder(pdfs_folder):
                summaries = []
                for pdf_file in pdfs_folder:
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(pdf_file.getvalue())
                    loader = PyPDFLoader(file_path=temp_path)
                    docs = loader.load_and_split()
                    chain = load_summarize_chain(llm, chain_type="map_reduce")
                    summary = chain.run(docs)
                    summaries.append(summary)
                    os.remove(temp_path)
                return summaries

            summaries = summarize_pdfs_from_folder([uploaded_file])
            for summary in summaries:
                st.write(summary)
