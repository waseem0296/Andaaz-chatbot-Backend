
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

import pytesseract
from PIL import Image
from io import BytesIO


load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def extract_text_from_image(img_url):
    try:
        # Only process PNG, JPG, JPEG
        if not (img_url.lower().endswith(".png") or img_url.lower().endswith(".jpg") or img_url.lower().endswith(".jpeg")):
            print(f"Skipping unsupported image format: {img_url}")
            return ""

        response = requests.get(img_url, timeout=10)
        img = Image.open(BytesIO(response.content))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Image OCR failed for {img_url}: {e}")
        return ""


# --- Website Crawler + Scraper
def scrape_site_all_pages(base_url, max_pages=30):
    visited = set()
    to_visit = [base_url]
    all_docs = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        print(f"Scraping: {current_url}") 

        try:
            response = requests.get(current_url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {current_url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()


        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        full_text = "\n".join(lines)

        # --- Extract text from images ---   # ⬅️ INSERTED HERE
        image_texts = []
        for img_tag in soup.find_all("img", src=True):
            img_url = urljoin(current_url, img_tag["src"])
            ocr_text = extract_text_from_image(img_url)
            if ocr_text:
                image_texts.append(ocr_text)

        # Combine HTML text + OCR text
        full_text = full_text + "\n".join(image_texts)

        all_docs.append(Document(page_content=full_text, metadata={"source": current_url}))

        visited.add(current_url)

        for link_tag in soup.find_all("a", href=True):
            href = link_tag["href"]
            joined_url = urljoin(base_url, href)
            parsed_base = urlparse(base_url)
            parsed_joined = urlparse(joined_url)

            if parsed_base.netloc == parsed_joined.netloc and joined_url not in visited:
                to_visit.append(joined_url)

    return all_docs

# --- Step 1: Scrape website and extract content ---
WEBSITE_URL = "https://demoweb.andaaz.es/"
raw_docs = scrape_site_all_pages(WEBSITE_URL, max_pages=25)  

# --- Step 2: Split content into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# Limit documents to prevent embedding overload
MAX_DOCS = 100
if len(docs) > MAX_DOCS:
    docs = docs[:MAX_DOCS]

# --- Step 3: Generate vector embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# --- Step 4: Define custom prompt ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI chatbot that is made for support of customers related to questions of this website "https://demoweb.andaaz.es/"
and you will answer atentic answer by using your knowledge and must use the content of this website and it is exacted also. Always remain one thing in mind never start answer with like 'Based on content of this site etc..', Your should be like a real human is talking.

Context:
{context}

Question:
{question}

Answer:
"""
)

# You are an AI chatbot that only answers questions based on the provided website content.
# If the question is not relevant to the website, reply: "I'm sorry, I don't know the answer to that based on this website."

# --- Step 5: Configure Groq LLM and QA pipeline ---
llm = ChatGroq(
    model_name="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

# --- Public Function: Called from Flask ---
def answer_query(question):
    result = qa.invoke({"query": question})
    return result["result"] if isinstance(result, dict) else result
