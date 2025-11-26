from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import tempfile
from datetime import datetime
import re

from main import (
    process_pdf,
    process_web,
    create_vector_store,
    get_rag_agent,
    get_web_search_agent,
    init_qdrant,
    check_document_relevance,
    OllamaEmbedderr
)

app = FastAPI(title="DeepSeek R1 RAG API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="static")

# Global state management (In production, use Redis or a database)
class AppState:
    def __init__(self):
        self.vector_store = None
        self.processed_documents = []
        self.qdrant_client = None
        self.rag_enabled = True
        self.use_web_search = False
        self.similarity_threshold = 0.7
        self.search_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
        self.chat_history = {}  # user_id -> messages

app_state = AppState()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: str
    force_web_search: bool = False
    rag_enabled: bool = True

class ChatResponse(BaseModel):
    response: str
    thinking_process: Optional[str] = None
    sources: Optional[List[dict]] = None
    search_type: Optional[str] = None

class ConfigRequest(BaseModel):
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    rag_enabled: bool = True
    use_web_search: bool = False
    similarity_threshold: float = 0.7
    search_domains: Optional[List[str]] = None

class DocumentResponse(BaseModel):
    success: bool
    message: str
    document_name: str

class HistoryResponse(BaseModel):
    history: List[dict]

# Root endpoint - serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Configuration endpoints
@app.post("/api/configure")
async def configure_system(config: ConfigRequest):
    """Configure Qdrant and other system settings"""
    try:
        if config.qdrant_url:
            app_state.qdrant_client = init_qdrant(
                config.qdrant_url, 
                config.qdrant_api_key
            )
            if not app_state.qdrant_client:
                raise HTTPException(status_code=400, detail="Failed to connect to Qdrant")
        
        app_state.rag_enabled = config.rag_enabled
        app_state.use_web_search = config.use_web_search
        app_state.similarity_threshold = config.similarity_threshold
        
        if config.search_domains:
            app_state.search_domains = config.search_domains
        
        return JSONResponse(content={
            "success": True,
            "message": "Configuration updated successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return JSONResponse(content={
        "rag_enabled": app_state.rag_enabled,
        "use_web_search": app_state.use_web_search,
        "similarity_threshold": app_state.similarity_threshold,
        "search_domains": app_state.search_domains,
        "has_qdrant": app_state.qdrant_client is not None,
        "processed_documents": app_state.processed_documents
    })

# Document upload endpoints
@app.post("/api/upload/pdf", response_model=DocumentResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    try:
        if not app_state.rag_enabled:
            raise HTTPException(status_code=400, detail="RAG mode is disabled")
        
        if not app_state.qdrant_client:
            raise HTTPException(status_code=400, detail="Qdrant not configured")
        
        if file.filename in app_state.processed_documents:
            return DocumentResponse(
                success=False,
                message="Document already processed",
                document_name=file.filename
            )
        
        # Read file content
        content = await file.read()
        
        # Process PDF
        texts = process_pdf(content, file.filename)
        
        if not texts:
            raise HTTPException(status_code=400, detail="Failed to process PDF")
        
        # Add to vector store
        if app_state.vector_store:
            app_state.vector_store.add_documents(texts)
        else:
            app_state.vector_store = create_vector_store(
                app_state.qdrant_client, 
                texts
            )
        
        app_state.processed_documents.append(file.filename)
        
        return DocumentResponse(
            success=True,
            message="PDF processed successfully",
            document_name=file.filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/url", response_model=DocumentResponse)
async def upload_url(url: str = Form(...)):
    """Process and add content from a URL"""
    try:
        if not app_state.rag_enabled:
            raise HTTPException(status_code=400, detail="RAG mode is disabled")
        
        if not app_state.qdrant_client:
            raise HTTPException(status_code=400, detail="Qdrant not configured")
        
        if url in app_state.processed_documents:
            return DocumentResponse(
                success=False,
                message="URL already processed",
                document_name=url
            )
        
        # Process URL
        texts = process_web(url)
        
        if not texts:
            raise HTTPException(status_code=400, detail="Failed to process URL")
        
        # Add to vector store
        if app_state.vector_store:
            app_state.vector_store.add_documents(texts)
        else:
            app_state.vector_store = create_vector_store(
                app_state.qdrant_client, 
                texts
            )
        
        app_state.processed_documents.append(url)
        
        return DocumentResponse(
            success=True,
            message="URL processed successfully",
            document_name=url
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/clear")
async def clear_documents():
    """Clear all processed documents"""
    try:
        app_state.processed_documents = []
        app_state.vector_store = None
        
        return JSONResponse(content={
            "success": True,
            "message": "All documents cleared"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return response"""
    try:
        user_id = request.user_id
        message = request.message
        force_web_search = request.force_web_search
        rag_enabled = request.rag_enabled
        
        # Initialize chat history for user if not exists
        if user_id not in app_state.chat_history:
            app_state.chat_history[user_id] = []
        
        # Add user message to history
        app_state.chat_history[user_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        context = ""
        docs = []
        search_type = None
        
        if rag_enabled and app_state.rag_enabled:
            # RAG mode processing
            if not force_web_search and app_state.vector_store:
                # Try document search first
                retriever = app_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": 5,
                        "score_threshold": app_state.similarity_threshold
                    }
                )
                docs = retriever.invoke(message)
                
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    search_type = "document"
            
            # Use web search if forced or no documents found
            if (force_web_search or not context) and app_state.use_web_search:
                try:
                    web_search_agent = get_web_search_agent(app_state.search_domains)
                    web_results = web_search_agent.run(message).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        search_type = "web"
                except Exception as e:
                    print(f"Web search error: {str(e)}")
            
            # Generate response with RAG agent
            rag_agent = get_rag_agent()
            
            if context:
                full_prompt = f"""Context: {context}

Original Question: {message}
Please provide a comprehensive answer based on the available information."""
            else:
                full_prompt = f"Original Question: {message}\n"
            
            response = rag_agent.run(full_prompt)
            response_content = response.content
            
        else:
            # Simple mode without RAG
            rag_agent = get_rag_agent()
            
            # Handle web search if forced
            if force_web_search and app_state.use_web_search:
                try:
                    web_search_agent = get_web_search_agent(app_state.search_domains)
                    web_results = web_search_agent.run(message).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        search_type = "web"
                except Exception as e:
                    print(f"Web search error: {str(e)}")
            
            if context:
                full_prompt = f"""Context: {context}

Question: {message}

Please provide a comprehensive answer based on the available information."""
            else:
                full_prompt = message
            
            response = rag_agent.run(full_prompt)
            response_content = response.content
        
        # Extract thinking process
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response_content, re.DOTALL)
        
        thinking_process = None
        if think_match:
            thinking_process = think_match.group(1).strip()
            final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
        else:
            final_response = response_content
        
        # Prepare sources
        sources = None
        if docs and search_type == "document":
            sources = []
            for i, doc in enumerate(docs, 1):
                source_type = doc.metadata.get("source_type", "unknown")
                source_name = doc.metadata.get(
                    "file_name" if source_type == "pdf" else "url",
                    "unknown"
                )
                sources.append({
                    "index": i,
                    "type": source_type,
                    "name": source_name,
                    "content": doc.page_content[:200] + "..."
                })
        
        # Add assistant response to history
        app_state.chat_history[user_id].append({
            "role": "assistant",
            "content": final_response,
            "thinking_process": thinking_process,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=final_response,
            thinking_process=thinking_process,
            sources=sources,
            search_type=search_type
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{user_id}", response_model=HistoryResponse)
async def get_chat_history(user_id: str):
    """Get chat history for a user"""
    if user_id not in app_state.chat_history:
        return HistoryResponse(history=[])
    
    return HistoryResponse(history=app_state.chat_history[user_id])

@app.delete("/api/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user"""
    if user_id in app_state.chat_history:
        app_state.chat_history[user_id] = []
    
    return JSONResponse(content={
        "success": True,
        "message": "Chat history cleared"
    })

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
