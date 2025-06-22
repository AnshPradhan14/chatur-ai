import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Directory to store the ChromaDB database
CHROMA_PERSIST_DIRECTORY = "chroma_db"
# Model for generating embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents(file_path):
    """Loads text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
        print(f"Successfully extracted text from {file_path}")
        return text
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return None

def split_documents(text):
    """Splits text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helps in debugging/referencing original text
    )
    chunks = text_splitter.create_documents([text])
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def create_and_persist_vector_db(chunks):
    """Generates embeddings and stores them in ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Initialize the embedding model
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # Create and persist the ChromaDB vector store
    # If the directory exists, it will load the existing collection.
    # Otherwise, it will create a new one.
    print(f"Creating/loading ChromaDB at: {CHROMA_PERSIST_DIRECTORY}...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    db.persist() # Ensure data is written to disk
    print(f"ChromaDB created/updated with {len(chunks)} chunks.")
    return db

def main():
    # Example usage:
    # IMPORTANT: Place your sample PDF file in the same directory as this script,
    # or provide the full path to your PDF.
    pdf_file_name = "sample_notes.pdf" # <<< CHANGE THIS to your PDF filename

    # --- Create a dummy PDF for testing if you don't have one ---
    if not os.path.exists(pdf_file_name):
        print(f"'{pdf_file_name}' not found. Creating a dummy PDF for demonstration.")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_file_name)
        c.drawString(100, 750, "This is a sample document for testing.")
        c.drawString(100, 730, "It contains some basic information about AI and Machine Learning.")
        c.drawString(100, 710, "Machine learning is a subset of artificial intelligence.")
        c.drawString(100, 690, "It focuses on the development of computer programs that can access data and use it learn for themselves.")
        c.drawString(100, 670, "Deep learning is a specialized subset of machine learning that uses multi-layered neural networks.")
        c.drawString(100, 650, "Neural networks are inspired by the human brain and are designed to recognize patterns.")
        for i in range(10): # Add more content to make it chunkable
            c.drawString(100, 630 - i*10, f"Additional paragraph {i+1}: This expands on topics related to AI and its applications in various fields.")
            c.drawString(100, 610 - i*10, f"Further details on deep neural networks and their architecture. Page content {i+1}.")
        c.save()
        print(f"Dummy PDF '{pdf_file_name}' created. Please run the script again.")
        return
    # -----------------------------------------------------------

    # 1. Load Document
    raw_text = load_documents(pdf_file_name)
    if raw_text:
        # 2. Split Document into Chunks
        chunks = split_documents(raw_text)

        # 3. Create and Persist Vector Database
        db = create_and_persist_vector_db(chunks)
        
        print("\nData ingestion complete!")
        print(f"Vector database stored at: ./{CHROMA_PERSIST_DIRECTORY}")
        
        # Optional: Test retrieval (for verification)
        print("\nTesting retrieval from the database:")
        query_text = "What is machine learning?"
        results = db.similarity_search(query_text, k=2)
        print(f"\nQuery: '{query_text}'")
        print("Top 2 retrieved chunks:")
        for i, doc in enumerate(results):
            print(f"--- Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}, Index {doc.metadata.get('start_index', 'N/A')}) ---")
            print(doc.page_content)
            print("-" * 20)
    else:
        print("No text loaded. Aborting data ingestion.")

if __name__ == "__main__":
    main()