

"""
vector.py
---------
Builds a multi-document FAISS vector store.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st

from documents import DOCUMENTS

#  Convert Raw Documents
def create_documents():
    docs = []
    for doc in DOCUMENTS:
        docs.append(
            Document(
                page_content=doc["text"],   # keep text intact
                metadata={"source": doc["title"]}
            )
        )
    return docs


raw_documents = create_documents()

#  Chunk Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunked_documents = text_splitter.split_documents(raw_documents)

print(f"Total chunks created: {len(chunked_documents)}")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#  FAISS Vector Store
vector_store = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embedding_model
)

vector_store.save_local("faiss_index")
print("FAISS vector store built and saved successfully")


#  Retriever
@st.cache_resource(show_spinner="Loading FAISS retriever...")
def get_retriever(k: int = 3):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
