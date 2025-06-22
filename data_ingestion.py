# data_ingestion.py

import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
PERSIST_DIRECTORY = "db" # Directory to store your ChromaDB
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_and_chunk_pdf(file_path: str):
    """
    Loads text from a PDF file, splits it into chunks, and returns them.
    """
    print(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Initialize the text splitter
    # Recommended chunk size: 500-1000 characters with 100-200 character overlap.
    # We'll use 1000 characters and 200 overlap for a good balance.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks: list[str], persist_directory: str = PERSIST_DIRECTORY):
    """
    Generates embeddings for text chunks and stores them in a ChromaDB vector store.
    """
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Creating/Loading ChromaDB at: {persist_directory}")
    # This will create the embeddings and add them to the vector store
    # or load an existing one if it's already there.
    vector_store = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vector_store.persist() # Ensure data is written to disk
    print("Vector store created/updated and persisted.")
    return vector_store

if __name__ == "__main__":
    # --- Example Usage ---
    # Create a dummy PDF for testing if you don't have one readily available
    # For a real project, you'd have your actual study PDFs here.
    
    # Create a dummy text file first, then convert it to PDF if needed for testing
    dummy_text_content = """
    Artificial intelligence (AI) is a rapidly advancing field of computer science dedicated to creating machines that can perform tasks normally requiring human intelligence. This includes learning, problem-solving, pattern recognition, and understanding natural language.

    Machine learning (ML) is a subfield of AI that focuses on enabling systems to learn from data without explicit programming. It involves algorithms that build a model from sample data, known as "training data," in order to make predictions or decisions without being explicitly programmed to perform the task. Key concepts include supervised learning, unsupervised learning, and reinforcement learning.

    Deep learning (DL) is a specialized subfield of machine learning inspired by the structure and function of the human brain's neural networks. Deep learning algorithms use multiple layers to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers identify complex shapes and objects. Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are prominent architectures in deep learning.

    Chatbots are computer programs designed to simulate human conversation. AI-powered chatbots use natural language processing (NLP) to understand user queries and generate relevant responses. They can range from simple rule-based systems to complex models powered by Large Language Models (LLMs). Retrieval Augmented Generation (RAG) is a technique where an LLM retrieves information from a knowledge base to inform its answer, combining the strength of retrieval systems with the generative power of LLMs.
    """
    dummy_pdf_path = "dummy_study_notes.pdf"
    
    # Create a simple PDF for testing purposes.
    # In a real scenario, you would upload your actual study PDFs.
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        text_lines = dummy_text_content.split('\n')
        y_position = 750
        for line in text_lines:
            c.drawString(50, y_position, line.strip())
            y_position -= 12 # Move up for next line
            if y_position < 50: # New page if too low
                c.showPage()
                y_position = 750
        c.save()
        print(f"Created a dummy PDF: {dummy_pdf_path}")
    except ImportError:
        print("ReportLab not installed. Cannot create dummy PDF. Please provide a PDF manually or install ReportLab (`pip install reportlab`).")
        print("Proceeding assuming you will place a PDF named 'dummy_study_notes.pdf' in the current directory.")
        print("Alternatively, you can just use a .txt file and modify the code temporarily to read .txt instead of .pdf for testing.")
        # If ReportLab is not installed, you'd need to manually create `dummy_study_notes.pdf`
        # or temporarily change `PdfReader` to read a `.txt` file.

    if os.path.exists(dummy_pdf_path):
        # 1. Load and Chunk PDF
        chunks = load_and_chunk_pdf(dummy_pdf_path)

        # 2. Get Vector Store (create/load and persist)
        vector_store = get_vector_store(chunks)

        # You can test retrieval here if you like, for verification
        print("\n--- Testing Retrieval (After Ingestion) ---")
        query = "What is deep learning?"
        retrieved_docs = vector_store.similarity_search(query, k=2)
        print(f"Query: '{query}'")
        for i, doc in enumerate(retrieved_docs):
            print(f"Retrieved Chunk {i+1}:\n{doc.page_content[:200]}...\n")
    else:
        print("Please place a PDF file (e.g., 'my_study_material.pdf') in the same directory and update the `dummy_pdf_path` variable to its name for testing.")