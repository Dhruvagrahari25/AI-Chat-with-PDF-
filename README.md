Interactive PDF Chatbot with LLM

This project allows users to have a conversation with the content of a PDF document. Here's a breakdown of the technologies used:

* **Python:** The primary programming language for building the application logic.
* **FAISS (Facebook AI Similarity Search):** Enables efficient retrieval of relevant sections from the PDF based on user queries.
* **Streamlit:** Creates a user-friendly web interface for uploading PDFs and interacting with the chatbot.
* **Langchain:** Facilitates communication between the user and the LLM.
* **Huggingface Instruct Embedding function:** Processes user queries and generates instructions for the LLM to understand the context and respond accurately to the specific PDF content.

Essentially, the user uploads a PDF. FAISS indexes the document content. When the user asks a question, Langchain translates it and the Instruct Embedding function prepares instructions for the LLM. The LLM then searches the indexed PDF using FAISS and generates a response based on the retrieved content. Streamlit displays this response to the user, creating an interactive conversation-like experience with the PDF.
![Screenshot 2024-06-23 173742](https://github.com/user-attachments/assets/8b119089-7b88-46b6-8b6e-cf37787fbdae)
