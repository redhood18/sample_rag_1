api_token="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"

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import HfApi,login

from transformers import pipeline
import streamlit as st

login(token=api_token,add_to_git_credential=True)
api = HfApi()


def encode_pdf(file):
  loader = PyPDFLoader(path)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
  texts = text_splitter.split_documents(documents)
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  db = FAISS.from_documents(texts, embeddings)
  return db


pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B",  # Replace with the actual model ID
    torch_dtype="bfloat16",
    device_map="auto"
)
retriever=""
llm_model=HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B",task="text-generation",huggingfacehub_api_token=api_token,model_kwargs={"temperature": 0.1})
question="What are the causes for climate change?"
chat = f"""You are a knowledgeable assistant. Use the following context to answer the question:

Context:
{retriever}

Question:
{question}

Answer:
"""

response = pipe(chat, max_new_tokens=2000)

print(response[0]['generated_text'])

st.title("RAG")
file = st.file_uploader("Choose a (.pdf) file", type=["pdf"])

if file is not None:
    file=encode_pdf(file)

    retriever = file.as_retriever(search_kwargs={"k": 3})  # Top 5 relevant documents
    response = pipe(chat, max_new_tokens=2000)
    st.write(response[0]['generated_text'])

    
