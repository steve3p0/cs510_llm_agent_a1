"""
App:          RAG Application

Author:       Raghad Al Musawi
Date:         01/19/2026
Description:  This application demonstrates a simple Retrieval-Augmented Generation (RAG) system using
              LangChain and Google Generative AI. It loads documents from a specified URL, PDF file, 
              and DOCX file, chunks them for better retrieval, creates a vector database, and sets up 
              a RAG chain to answer user questions based on the loaded documents.
"""
import os
import readline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, Docx2txtLoader

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Get API key from environment (never hardcode it!)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")

llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL)


def load_documents(url, pdf_path, docx_path=None):
    """
    Load documents from a URL, PDF file, and optionally a DOCX file.
    """
    all_docs = []

    print(f"Loading URL: {url}")
    web_loader = WebBaseLoader(url)
    web_docs = web_loader.load()
    all_docs.extend(web_docs)

    print(f"Loading PDF: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    all_docs.extend(pdf_docs)

    if docx_path and os.path.exists(docx_path):
        print(f"Loading DOCX: {docx_path}")
        docx_loader = Docx2txtLoader(docx_path)
        docx_docs = docx_loader.load()
        all_docs.extend(docx_docs)

    print(f"Loaded {len(all_docs)} documents total")
    return all_docs


def chunk_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """
    Create a searchable vector database from document chunks.
    """
    print("Creating vector database...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_query"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./.chromadb"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Vector database created!")
    return retriever


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever):
    """
    Create the RAG chain using LCEL (LangChain Expression Language).
    """
    llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0)

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question using only the provided context. 
If you don't know the answer, say so. Keep answers concise.

Context: {context}"""),
        ("human", "{question}")
    ])

    # Create the chain using LCEL (like 09_rag_query.py)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    """Main application entry point."""
    print("=== Simple RAG Application ===\n")

    # Configuration
    URL = "https://huggingface.co/rag-datasets/" 
    ""
    PDF_PATH = "data/langchain_paper.pdf"
   # DOCX_PATH = "data/FamousQuotes.docx"

    # Step 1: Load documents
    docs = load_documents(URL, PDF_PATH, DOCX_PATH)

    # Step 2: Chunk them
    chunks = chunk_documents(docs)

    # Step 3: Create vector store
    retriever = create_vector_store(chunks)

    # Step 4: Create RAG chain
    rag_chain = create_rag_chain(retriever)

    # Step 5: Interactive query loop
    print("\n=== Ready to answer questions! ===")
    print("Type 'quit' to exit\n")

    # Show loaded document sources
    print("Documents loaded from:")
    document_data_sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        if 'source' in doc_metadata:
            document_data_sources.add(doc_metadata['source'])
    for doc in document_data_sources:
        print(f"  {doc}")
    print()

    while True:
        question = input("Question: ")
        if question.lower() == 'quit':
            break

        response = rag_chain.invoke(question)
        print(f"\nAnswer: {response}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
