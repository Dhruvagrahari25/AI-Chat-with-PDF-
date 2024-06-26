import streamlit as st
import PyPDF2 as PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import numpy as np
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader.PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks



# def get_vectorstore(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectorstore = FAISS.from_texts(text_chunks,embedding=embeddings)
#     vectorstore.save_local("faiss-index")
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks,embedding=embeddings)
    vectorstore.save_local("faiss-index")


def get_conversational_chain():
    Prompt_template="""
    Answer the question in as detail as possible from the provided context, make sure to provide all the details, if the answer to the question is not present in the given context just say, "answer not available in the given context", don't provide a wrong answer
    context:\n{context}?\n
    question: \n{question}\n
    
    Answer:
    """
    
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    
    prompt = PromptTemplate(template=Prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings()

    new_db =  FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True
    )
    
    print(response)
    st.write ("Reply: ", response["output_text"])
    


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

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
