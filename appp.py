import streamlit as st
import PyPDF2 as PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss-index")

def get_conversational_chain():
    prompt_template = """
    Answer the question in as much detail as possible from the provided context and previous conversation. Make sure to provide all the details. If the answer to the question is not present in the given context, just say, "answer not available in the given context", and don't provide a wrong answer.
    Previous conversations:\n{chat_history}\n
    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, memory):
    embeddings = HuggingFaceEmbeddings()
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chat_history = memory.load_memory_variables({})
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question, "chat_history": chat_history["chat_history"]},
        return_only_outputs=True
    )
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(response["output_text"])
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with multiple PDFs :books:")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, memory)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
