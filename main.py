import os
import tempfile
from datetime import datetime
from typing import List
import bs4
import ollama
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings


class OllamaEmbedderr(Embeddings):
    """Custom embeddings class for Ollama using snowflake-arctic-embed model"""
    
    def __init__(self, model_name="snowflake-arctic-embed"):
        """
        Initialize the OllamaEmbedderr with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
        """
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]


# Constants
COLLECTION_NAME = "deepseek-r1-rag"
MODEL_VERSION = "deepseek-r1:7b"
WEB_SEARCH_MODEL = "gpt-oss:20b"


def init_qdrant(qdrant_url: str, qdrant_api_key: str = None) -> QdrantClient | None:
    """
    Initialize Qdrant client with configured settings.
    
    Args:
        qdrant_url: The URL of the Qdrant instance
        qdrant_api_key: The API key for authentication (optional for local Qdrant)
    
    Returns:
        QdrantClient: The initialized Qdrant client if successful.
        None: If the initialization fails.
    """
    if not qdrant_url:
        return None
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )
        return client
    except Exception as e:
        print(f"Qdrant connection failed: {str(e)}")
        return None


def process_pdf(file_content: bytes, file_name: str) -> List:
    """
    Process PDF file content and add source metadata.
    
    Args:
        file_content: The binary content of the PDF file
        file_name: The name of the PDF file
    
    Returns:
        List of processed document chunks
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "pdf",
                "file_name": file_name,
                "timestamp": datetime.now().isoformat()
            })
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return text_splitter.split_documents(documents)
    
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """
    Process web URL and add source metadata.
    
    Args:
        url: The URL to process
    
    Returns:
        List of processed document chunks
    """
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    
    except Exception as e:
        print(f"Web processing error: {str(e)}")
        return []


def create_vector_store(client: QdrantClient, texts: List):
    """
    Create and initialize vector store with documents.
    
    Args:
        client: The Qdrant client instance
        texts: List of document chunks to add
    
    Returns:
        QdrantVectorStore instance or None on failure
    """
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
            print(f"Using existing collection: {COLLECTION_NAME}")
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=OllamaEmbedderr()
        )
        
        # Add documents
        vector_store.add_documents(texts)
        print("Documents stored successfully!")
        return vector_store
        
    except Exception as e:
        print(f"Vector store error: {str(e)}")
        return None


def get_web_search_agent(search_domains: List[str]) -> Agent:
    """
    Initialize a web search agent using GPT-OSS:20b model.
    
    Args:
        search_domains: List of domains to search from
    
    Returns:
        Agent instance configured for web search
    """
    return Agent(
        name="Web Search Agent",
        model=Ollama(id=WEB_SEARCH_MODEL),
        instructions=f"""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Focus on these domains: {', '.join(search_domains)}
        3. Compile and summarize the most relevant information
        4. Include sources in your response
        5. Prioritize recent and authoritative sources
        
        Format your response with:
        - Key findings
        - Source citations
        - Relevant details
        """,
        markdown=True,
        debug_mode=False
    )


def get_rag_agent() -> Agent:
    """
    Initialize the main RAG agent using DeepSeek-R1:7b model.
    
    Returns:
        Agent instance configured for RAG tasks
    """
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=MODEL_VERSION),
        instructions="""You are an Intelligent Agent specializing in providing accurate, well-reasoned answers.

        Your capabilities:
        - Deep reasoning and analysis
        - Context-aware responses
        - Clear and concise communication
        - Source attribution when using provided context
        
        When asked a question:
        - Analyze the question thoroughly
        - Use your reasoning capabilities to provide accurate answers
        - Be precise and cite specific details when context is provided
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        - Maintain accuracy and relevance
        
        When given web search results:
        - Clearly indicate that information comes from web search
        - Synthesize the information clearly
        - Attribute sources appropriately
        
        Always:
        - Show your reasoning process when helpful
        - Maintain high accuracy and clarity
        - Be honest about uncertainties
        - Provide actionable insights
        """,
        markdown=True,
        debug_mode=False
    )


def check_document_relevance(
    query: str,
    vector_store,
    threshold: float = 0.7
) -> tuple[bool, List]:
    """
    Check if there are relevant documents for a query.
    
    Args:
        query: The search query
        vector_store: The vector store to search
        threshold: Minimum similarity score threshold
    
    Returns:
        Tuple of (has_relevant_docs: bool, documents: List)
    """
    if not vector_store:
        return False, []
    
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": threshold}
        )
        docs = retriever.invoke(query)
        return bool(docs), docs
    except Exception as e:
        print(f"Error checking document relevance: {str(e)}")
        return False, []


def format_chat_history(history: List[dict], max_messages: int = 10) -> str:
    """
    Format chat history for context.
    
    Args:
        history: List of chat messages
        max_messages: Maximum number of messages to include
    
    Returns:
        Formatted chat history string
    """
    if not history:
        return ""
    
    # Get last N messages
    recent_history = history[-max_messages:]
    
    formatted = "Chat History:\n"
    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted += f"{role.capitalize()}: {content}\n"
    
    return formatted


if __name__ == "__main__":
    print("This module contains utility functions for the RAG system.")
    print(f"Model: {MODEL_VERSION}")
    print(f"Web Search Model: {WEB_SEARCH_MODEL}")
    print(f"Collection: {COLLECTION_NAME}")




    # $env:USER_AGENT = "RAG-Agent/1.0 (Python RAG Application)"
    # $env:USER_AGENT
