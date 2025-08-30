

# Ai_logic.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Load prebuilt vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI chatbot designed to provide *accurate and human-like answers* to questions about the website "https://demoweb.andaaz.es/".

Use the provided context to answer the user's question. Answer naturally, as if you are a real human. 
Do NOT start your answers with phrases like "Based on the website" or "According to the content".

- If the answer can be found in the context, respond clearly and directly.
- If the answer is *not in the context*, reply: "I'm sorry, I don't know the answer to that based on this website."
Context:
{context}

Question:
{question}

Answer:
"""
)

# LLM and QA
llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

def answer_query(question):
    result = qa.invoke({"query": question})
    return result["result"] if isinstance(result, dict) else result
