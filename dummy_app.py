import streamlit as st
import os
import shutil
import time
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

# ADD THESE TWO LINES AT THE VERY TOP, AFTER IMPORTS
from dotenv import load_dotenv
load_dotenv() # This line loads the variables from .env into os.environ

# --- Configuration Constants ---
# Directory for persisting the ChromaDB vector store
CHROMA_PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY" # This is the correct environment variable NAME
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 4

# --- Session State Initialization ---
# Initialize Streamlit session state variables for persistence across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_url_processed" not in st.session_state:
    st.session_state.last_url_processed = None
if "search_scope_file" not in st.session_state:
    st.session_state.search_scope_file = True
if "search_scope_url" not in st.session_state:
    st.session_state.search_scope_url = False
if "search_scope_ai" not in st.session_state:
    st.session_state.search_scope_ai = False

# --- Helper Functions ---

def set_page_defaults():
    """Configures Streamlit page settings like title, icon, and layout."""
    st.set_page_config(page_title="AI Study Buddy Chatbot", page_icon="📚", layout="wide")
    st.title("📚 AI Study Buddy Chatbot")
    st.markdown("Ask questions about your uploaded notes or web content!")
    st.info("💡 You can either upload PDF/TXT files using the sidebar, or provide a URL of a web page to expand the knowledge base.")


@st.cache_resource
def load_components():
    """
    Loads and caches expensive components like the embedding model,
    ChromaDB vector store, and the Groq Chat model.
    These components are loaded only once for performance.
    """
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings
        )

        groq_api_key = os.getenv(GROQ_API_KEY_ENV_VAR)
        if not groq_api_key:
            st.error(f"Error: Groq API Key not found. Please set the '{GROQ_API_KEY_ENV_VAR}' environment variable, either in your OS or in a .env file.")
            st.stop()

        llm = ChatGroq(model_name=GROQ_MODEL_NAME, groq_api_key=groq_api_key)
        
        # Test Groq connectivity (optional but good practice)
        try:
            llm.invoke("Hello")
            # REMOVED: st.success("Successfully connected to Groq API.") from here
            # Instead, we'll set a flag to display it once outside the cached function.
        except Exception as groq_e:
            st.error(f"Failed to connect to Groq API with model '{GROQ_MODEL_NAME}': {groq_e}. Please check your API key and model name.")
            st.stop()

        return embeddings, vector_db, llm
    except Exception as e:
        st.error(f"Error loading core components: {e}")
        st.stop()

def get_document_text(uploaded_file):
    """
    Extracts text content from an uploaded PDF or TXT file.
    Args:
        uploaded_file: Streamlit UploadedFile object.
    Returns:
        str: Extracted text content.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    if file_extension == ".pdf":
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or "" # Handle pages with no extractable text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    elif file_extension == ".txt":
        try:
            # Decode as UTF-8, with error handling for common issues
            text = uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
            return None
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        return None
    return text

def get_web_content(url):
    """
    Extracts text content from a given URL using WebBaseLoader.
    Args:
        url (str): The URL to fetch content from.
    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading content from URL: {e}")
        return []

def process_documents(docs_to_process, vector_db_instance, source_type="uploaded_file"):
    """
    Splits documents into chunks, generates embeddings, and adds them to ChromaDB.
    Args:
        docs_to_process (list[Document]): List of LangChain Document objects.
        vector_db_instance: The ChromaDB vector store instance.
        source_type (str): 'uploaded_file' or 'url' for metadata tagging.
    """
    if not docs_to_process:
        st.warning("No new content to process.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    # Ensure source metadata is captured for filtering later
    processed_chunks = []
    for doc in docs_to_process:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            # Ensure a 'source' key exists, defaulting if not provided
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = doc.metadata.get('source', 'Unknown Source')
            # Custom tag to distinguish between file uploads and URLs for filtering
            chunk.metadata['original_source_type'] = source_type
            processed_chunks.append(chunk)

    if processed_chunks:
        with st.spinner("Adding to knowledge base... This might take a moment."):
            try:
                # Add chunks to the vector database
                vector_db_instance.add_documents(processed_chunks)
                vector_db_instance.persist() # Save changes to disk
                st.success(f"Successfully added {len(processed_chunks)} chunks to the knowledge base!")
            except Exception as e:
                st.error(f"Failed to add documents to vector store: {e}")
    else:
        st.warning("No chunks were generated from the provided content.")

def get_rag_chain_prompt(use_ai_knowledge: bool):
    """
    Constructs and returns the dynamic RAG prompt based on whether AI general knowledge is allowed.
    Args:
        use_ai_knowledge (bool): If True, allows the AI to use its general knowledge.
                                 If False, strictly confines it to provided context.
    Returns:
        ChatPromptTemplate: The configured RAG prompt.
    """
    # Base instructions for the AI assistant
    base_instructions = """
    You are an expert AI assistant specializing in educational content, designed to answer user queries comprehensively and accurately by leveraging provided context. Your primary goal is to synthesize information from the retrieved student notes to formulate a clear, concise, and satisfying answer.

    **Instruction Guidelines:**

    1.  **Role and Tone:** Adopt the persona of a knowledgeable and patient teacher. Your tone should be authoritative yet encouraging, similar to an educator explaining complex concepts to a curious student. Your explanations should be thorough, insightful, and easy to understand.

    2.  **Answer Length and Depth:**
        * **Dynamically Adjust Length:** The length of your answer should directly correspond to the complexity and depth required by the user's question and the richness of the provided context.
            * For simple, factual queries with direct answers in the context, a concise and direct response is appropriate.
            * For complex questions requiring synthesis, explanation, or comparison, provide a more elaborate and detailed answer, ensuring all facets of the question are addressed using the context.
        * **Completeness:** Ensure the answer feels "satisfying" by covering all relevant aspects of the question that the context allows, leaving the user with a comprehensive understanding. Avoid abrupt or overly brief responses if more information is available and pertinent.

    3.  **Structure and Clarity:**
        * Organize your answer logically using clear paragraphs, and use **bullet points or numbered lists** when presenting multiple distinct pieces of information or steps, if it enhances readability.
        * Start with a direct answer or a summary, then elaborate with supporting details drawn *directly* from the context.
        * Use clear, precise language. Avoid jargon where simpler terms suffice, or briefly explain technical terms if they are essential and appear in the context.
    """

    # Contextual reliance instructions, changes based on 'use_ai_knowledge'
    context_reliance_instructions = ""
    if use_ai_knowledge:
        context_reliance_instructions = """
    4.  **Contextual Reliance (Flexible with AI Knowledge):**
        * **Prioritize Provided Context:** Your answers should primarily be based on the `Relevant notes from uploaded documents:` provided below.
        * **Supplement with General Knowledge:** If the provided notes do not contain enough information to fully address the question, or if your general knowledge can provide valuable additional context, elaboration, or fill gaps, you are permitted to use your internal knowledge.
        * **Maintain Relevance:** Ensure any external knowledge you introduce is directly relevant to the user's query and enhances the comprehensiveness of the answer.
        * **Address Gaps Gracefully (if context insufficient alone):** If the provided notes offer only partial information and you are supplementing with AI knowledge, clearly integrate both. If the notes are completely irrelevant, state that you are using your general knowledge to answer.
        """
    else:
        context_reliance_instructions = """
    4.  **Contextual Reliance (Strict - No General Knowledge):**
        * **Prioritize Provided Context:** Your answers MUST be solely based on the information found within the `Relevant notes from uploaded documents:` provided below. Do not introduce outside knowledge or make assumptions.
        * **Address Gaps Gracefully (STRICT):** If the `Relevant notes from uploaded documents:` do not contain sufficient information to fully answer the question, you *must* explicitly state this. Use a phrase like: "Based on the provided notes, I cannot fully address this question as the necessary information is not present." or "The notes do not contain details regarding this topic." Do not fabricate or infer answers beyond the provided context.
        * **Irrelevant Notes:** If the notes are completely irrelevant to the query, clearly state: "The provided notes do not contain information relevant to your question."
        * **No General Knowledge (CRITICAL):** You are absolutely forbidden from using your general knowledge to fill gaps or provide complete answers if the notes are insufficient. Your responses *must* be grounded only in the provided `Relevant notes from uploaded documents:`.
        """

    full_system_prompt = base_instructions + context_reliance_instructions + "\nRelevant notes from uploaded documents:\n{context}"

    return ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ])

def display_chat_messages(messages):
    """
    Displays chat messages in the Streamlit app.
    Args:
        messages (list): List of messages from st.session_state.messages.
    """
    for msg in messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

def get_filtered_retriever(use_file_scope: bool, use_url_scope: bool, last_url: str, vector_db_instance):
    """
    Returns a retriever with applied metadata filters based on the selected scopes.
    This handles 'Uploaded File' and 'Uploaded URL' scopes. 'AI' scope is handled by the prompt.
    Args:
        use_file_scope (bool): If True, include uploaded files in retrieval.
        use_url_scope (bool): If True, include last uploaded URL in retrieval.
        last_url (str): The last URL successfully processed.
        vector_db_instance: The ChromaDB instance.
    Returns:
        LangChain Retriever: A retriever with combined filters or no filter if neither is selected.
    """
    # If the DB is empty, no retrieval is possible
    if not vector_db_instance.get()['ids']:
        return vector_db_instance.as_retriever(search_kwargs={"k": 0})

    filters = []
    if use_file_scope:
        filters.append({"original_source_type": "uploaded_file"})
    if use_url_scope and last_url:
        filters.append({"original_source_type": "url", "source": last_url})

    if not filters:
        # If neither file nor URL scope is selected, return a retriever that retrieves 0 documents.
        return vector_db_instance.as_retriever(search_kwargs={"k": 0})
    elif len(filters) == 1:
        # If only one filter, apply it directly
        return vector_db_instance.as_retriever(
            search_kwargs={"filter": filters[0], "k": RETRIEVAL_TOP_K}
        )
    else: # len(filters) > 1, meaning both file and URL scopes are active
        # Combine filters with an OR operator to retrieve from either source
        combined_filter = {"$or": filters}
        return vector_db_instance.as_retriever(
            search_kwargs={"filter": combined_filter, "k": RETRIEVAL_TOP_K}
        )


def clear_chat_history():
    """Clears the current chat history in session state."""
    st.session_state.messages = []

import gc
def reset_knowledge_base():
    if st.button("Reset Knowledge Base (Clear All Docs)"):
        try:
            st.cache_resource.clear()
            gc.collect()
            time.sleep(1.5)

            if os.path.exists(CHROMA_PERSIST_DIRECTORY):
                shutil.rmtree(CHROMA_PERSIST_DIRECTORY)

            st.success("Knowledge base will be cleared after restart.")
            st.markdown("🔄 Please **manually reload the page** or close and reopen the Streamlit app.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Could not clear knowledge base: {e}")

def sidebar_ui(vector_db_instance): # Pass vector_db_instance here
    """Renders the sidebar UI elements for document management and chat options."""
    with st.sidebar:
        st.header("Manage Knowledge Base")

        st.markdown(
            "***Tip: To change the app's theme (e.g., to Dark Mode by default), create "
            "a `.streamlit/config.toml` file in your app's directory with `[theme]` "
            "settings. For example: `[theme]\nbase=\"dark\"`***"
        )
        st.markdown("---") # Visual separator

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded_files:
            documents_to_add = []
            for uploaded_file in uploaded_files:
                file_content = get_document_text(uploaded_file)
                if file_content:
                    documents_to_add.append(Document(page_content=file_content, metadata={"source": uploaded_file.name}))
            process_documents(documents_to_add, vector_db_instance, source_type="uploaded_file")
            st.session_state.last_url_processed = None # Clear last URL if new files are uploaded
            # REMOVED: st.session_state.current_search_scope = "All Documents" # THIS LINE IS REMOVED


        st.subheader("Add Web Content (URL)")
        url_input = st.text_input("Enter a URL", key="url_input")
        if st.button("Add URL to Knowledge Base", use_container_width=True):
            if url_input:
                with st.spinner(f"Fetching content from {url_input}..."):
                    web_docs = get_web_content(url_input)
                    if web_docs:
                        process_documents(web_docs, vector_db_instance, source_type="url")
                        st.session_state.last_url_processed = url_input # Store for scope filtering
                        # REMOVED: st.session_state.current_search_scope = "Recently Added URL" # THIS LINE IS REMOVED
                    else:
                        st.warning("Could not fetch content from the provided URL.")
            else:
                st.warning("Please enter a URL.")

        st.markdown("---") # Separator for better UI organization

        st.subheader("Search Scope Selection")
        # Checkboxes for search scope
        st.session_state.search_scope_file = st.checkbox(
            "Uploaded File",
            value=st.session_state.search_scope_file,
            help="Include content from your uploaded PDF/TXT files.",
            key="scope_file_checkbox"
        )
        st.session_state.search_scope_url = st.checkbox(
            "Uploaded URL",
            value=st.session_state.search_scope_url,
            help="Include content from the last provided URL.",
            disabled=not st.session_state.last_url_processed, # Disable if no URL processed
            key="scope_url_checkbox"
        )
        st.session_state.search_scope_ai = st.checkbox(
            "AI (General Knowledge)",
            value=st.session_state.search_scope_ai,
            help="Allow the AI to use its internal knowledge in addition to documents.",
            key="scope_ai_checkbox"
        )

        # Ensure at least one scope is selected if the DB is not empty
        num_docs_in_db = len(vector_db_instance.get()['ids'])
        if num_docs_in_db > 0 and not (st.session_state.search_scope_file or st.session_state.search_scope_url or st.session_state.search_scope_ai):
            st.warning("Please select at least one search scope.")


        st.markdown("---") # Separator

        # Buttons for chat history and knowledge base management
        st.button(
            "Clear Chat History",
            on_click=clear_chat_history,
            use_container_width=True,
            help="Erase the current conversation.",
        )
        import gc
        import time

        if st.button("Reset Knowledge Base (Clear All Docs)"):
            try:
                # Attempt to delete cached vector DB reference
                if "vector_db" in globals():
                    del globals()["vector_db"]
                if "vector_db" in st.session_state:
                    del st.session_state["vector_db"]

                st.cache_resource.clear()
                gc.collect()
                time.sleep(1)  # Give the OS a moment to release file locks

                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
                            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
                        break
                    except PermissionError as e:
                        if attempt == max_retries - 1:
                            st.error(f"❌ Failed to clear knowledge base after {max_retries} attempts: {e}")
                            st.stop()
                        time.sleep(0.8)

                # Reset Streamlit states
                st.session_state.processed_files = []
                st.session_state.processed_urls = []
                st.session_state.last_processed_url = None
                st.session_state.current_search_scope = "All Documents"
                st.session_state.messages = []

                st.success("Knowledge base successfully cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Unhandled error during reset: {e}")


        # Display current ChromaDB status
        st.caption(f"Documents in DB: {num_docs_in_db}")

def main_chat_interface(llm_model, vector_db_instance):
    """Manages the main chat display and user interaction."""
    # Display existing messages from session state
    display_chat_messages(st.session_state.messages)

    # User input for new query
    user_query = st.chat_input("Ask a question about your study material...")

    if user_query:
        # Add user query to chat history
        st.session_state.messages.append(HumanMessage(content=user_query, type="human"))
        # Rerun to display the user's message immediately and proceed to AI response
        st.rerun()

    # Process AI response only if the last message in history was from the user
    if st.session_state.messages and st.session_state.messages[-1].type == "human":
        with st.chat_message("ai"): # Display AI's thinking process
            with st.spinner("Thinking..."):
                ai_response_content = ""
                retrieved_sources = []
                retrieved_docs_found = False # Flag to track if any docs were actually retrieved

                # Determine if AI general knowledge should be used
                use_ai_knowledge = st.session_state.search_scope_ai

                # Dynamically get the RAG prompt based on AI knowledge scope
                rag_prompt_instance = get_rag_chain_prompt(use_ai_knowledge)
                # Ensure chat_history_for_llm is correctly constructed
                chat_history_for_llm = [
                    HumanMessage(content=msg.content) if isinstance(msg, HumanMessage) else AIMessage(content=msg.content)
                    for msg in st.session_state.messages[:-1] # Exclude the current human query as it's passed separately
                ]
                rag_chain = rag_prompt_instance | llm_model

                # Check if any document-based scope is selected AND there are documents in DB
                has_docs_in_db = len(vector_db_instance.get()['ids']) > 0
                search_docs_enabled = st.session_state.search_scope_file or (st.session_state.search_scope_url and st.session_state.last_url_processed)

                context_content = "" # Initialize context, will be populated if docs are retrieved

                if search_docs_enabled and has_docs_in_db:
                    try:
                        # Get the retriever with the applied filter based on the current scope
                        active_retriever = get_filtered_retriever(
                            st.session_state.search_scope_file,
                            st.session_state.search_scope_url,
                            st.session_state.last_url_processed,
                            vector_db_instance
                        )

                        # Retrieve relevant documents for the query
                        retrieved_docs = active_retriever.invoke(user_query)
                        if retrieved_docs:
                            context_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            retrieved_docs_found = True

                            # Extract unique sources from retrieved documents for display
                            unique_sources = set()
                            for doc in retrieved_docs:
                                source_name = doc.metadata.get("source", "Unknown Source")
                                # Enhance source display with title for URLs and page for PDFs
                                if "title" in doc.metadata and doc.metadata.get("original_source_type") == "url":
                                    source_name = f"{doc.metadata['title']} ({source_name})"
                                elif "page" in doc.metadata and doc.metadata.get("original_source_type") == "uploaded_file":
                                    source_name = f"{source_name} (Page {doc.metadata['page']})"
                                unique_sources.add(source_name)
                            retrieved_sources = sorted(list(unique_sources))

                    except Exception as e:
                        st.error(f"Error during document retrieval: {e}")
                        context_content = "" # Clear context if retrieval failed
                        retrieved_sources = []
                        retrieved_docs_found = False
                elif not has_docs_in_db and (st.session_state.search_scope_file or st.session_state.search_scope_url):
                    # User selected file/URL scope but no docs are in DB
                    # If AI knowledge is NOT also selected, warn the user and stop
                    if not use_ai_knowledge:
                         ai_response_content = "No documents have been added to the knowledge base to search within the selected document scopes ('Uploaded File', 'Uploaded URL'). Please upload files or add a URL, or enable 'AI (General Knowledge)' to get an answer."
                         st.markdown(ai_response_content)
                         st.session_state.messages.append(AIMessage(content=ai_response_content, type="ai"))
                         st.rerun() # Stop processing and display warning
                         return # Exit the function

                # If no document-based context was found/requested AND AI general knowledge is NOT selected
                if not retrieved_docs_found and not use_ai_knowledge:
                     ai_response_content = "No relevant information found within the selected scopes. Please upload relevant documents or enable 'AI (General Knowledge)' for broader answers."
                     st.markdown(ai_response_content)
                     st.session_state.messages.append(AIMessage(content=ai_response_content, type="ai"))
                     st.rerun()
                     return # Exit the function if no sources are active and no AI knowledge

                # Invoke the RAG chain to get the AI's response
                try:
                    ai_response_obj = rag_chain.invoke({
                        "context": context_content, # Pass retrieved context (could be empty if no docs/no retrieval)
                        "chat_history": chat_history_for_llm,
                        "query": user_query
                    })
                    ai_response_content = ai_response_obj.content
                except Exception as e:
                    ai_response_content = (f"An error occurred while generating the response: {e}. "
                                           "Please ensure your Groq API key is correct and the Groq service is accessible.") # Updated error message
                    retrieved_sources = [] # No sources if LLM invocation failed

            # Display the AI's generated response
            st.markdown(ai_response_content)

            # Display sources if available, in an expander
            if retrieved_sources:
                with st.expander("Sources Used:"):
                    for source in retrieved_sources:
                        st.markdown(f"- {source}")
            elif (st.session_state.search_scope_file or st.session_state.search_scope_url) and not retrieved_docs_found:
                 with st.expander("Sources Used:"):
                    st.markdown(f"No direct source material highly relevant to this specific question was found within the selected document scopes.")
            elif not has_docs_in_db: # If no documents at all
                with st.expander("Sources Used:"):
                    st.markdown("No documents or URLs have been added to the knowledge base yet.")


        # Add the AI's response to the session state chat history
        st.session_state.messages.append(AIMessage(content=ai_response_content, type="ai"))
        # Rerun to clear the chat input box for the next query
        st.rerun()


# --- Application Entry Point ---
if __name__ == "__main__":
    # Initialize Streamlit page and common components
    set_page_defaults()

    # Load shared AI/DB resources (cached)
    embeddings_model, vector_db_instance, llm_model = load_components()

    # Render sidebar UI, passing the vector_db_instance
    sidebar_ui(vector_db_instance)

    # Render main chat interface, passing llm_model and vector_db_instance
    main_chat_interface(llm_model, vector_db_instance)