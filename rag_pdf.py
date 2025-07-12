import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

def read_pdf(pdf_path):
    """
    Reads a PDF file.
    """
    print("Starting the process to read PDF...")
    
    print("Loading PDF and creating vector store...")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from the PDF.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(
        documents=docs,
    )
    return chunks

persist_directory = 'db'

def create_vector_store(chunks, persist_directory, EMBEDDING_MODEL, COLLECTION_NAME):
    """
    Creates a vector store from the chunks.
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    print(f"Creating vector store in {persist_directory}...")
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )
    print("Finalizing vector store...")

    return vector_store
