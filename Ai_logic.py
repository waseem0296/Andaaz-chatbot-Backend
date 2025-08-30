

# Ai_logic.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Optimized and efficient custom embedding class
class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        """
        Initializes the embedding model and tokenizer.
        
        Args:
            model_name (str): The name of the pre-trained model on Hugging Face Hub.
            device (str, optional): The device to run the model on ('cuda' for GPU, 'cpu'). 
                                    Defaults to None, which auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling over the token embeddings, ignoring padding tokens.
        
        This is the standard and efficient method used by the sentence-transformers library.
        
        Args:
            model_output: The output from the transformer model.
            attention_mask: The attention mask from the tokenizer.
        
        Returns:
            torch.Tensor: The sentence embeddings.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state
        
        # Expand attention mask to match token embeddings size
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Multiply embeddings by the expanded mask to zero out padding tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the mask values to get the number of non-padding tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Divide the sum of embeddings by the sum of mask values to get the average
        mean_pooled = sum_embeddings / sum_mask
        
        # Normalize the embeddings to unit vectors for cosine similarity
        return F.normalize(mean_pooled, p=2, dim=1)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.
        
        Args:
            texts (List[str]): The list of texts to embed.
        
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        # Tokenize the input texts
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad(): # Disable gradient calculation for inference to save memory and speed up
            model_output = self.model(**encoded_input)

        # Apply mean pooling and convert to a list of lists
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query.
        
        Args:
            text (str): The query text to embed.
        
        Returns:
            List[float]: The query embedding as a list of floats.
        """
        return self.embed_documents([text])[0]

# Now, use your custom embedding class in your main script
embedding_model = CustomHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Your existing code for loading the vectorstore and running the QA chain
# Note: The FAISS index must have been created using the same embedding model.
# If you are recreating the index, this will download the model weights (~80 MB).
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