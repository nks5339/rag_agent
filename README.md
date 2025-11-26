# DeepSeek R1 RAG Agent - FastAPI Implementation

A powerful RAG (Retrieval-Augmented Generation) system using DeepSeek-R1:7b model with FastAPI backend and modern web interface.

## Features

- 🤖 **DeepSeek-R1:7b** - Advanced reasoning LLM
- 📚 **RAG Support** - Context-aware responses using document retrieval
- 🌐 **Web Search** - GPT-OSS:20b powered web search integration
- 🗂️ **Document Processing** - PDF and web URL support
- 💾 **Vector Store** - Open-source Qdrant database
- 🎨 **Modern UI** - Clean, responsive web interface
- 💭 **Thinking Visualization** - See the reasoning process
- 📊 **Source Attribution** - Track document sources

## Prerequisites

### 1. Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### 2. Pull Required Models

```bash
# Main reasoning model
ollama pull deepseek-r1:7b

# Web search model
ollama pull gpt-oss:20b

# Embedding model
ollama pull snowflake-arctic-embed
```

### 3. Setup Qdrant (Choose One Option)

#### Option A: Local Qdrant (Docker)
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

#### Option B: Qdrant Cloud (Free Tier)
1. Sign up at https://cloud.qdrant.io
2. Create a cluster
3. Get your API key and URL

## Installation

1. **Clone or Create Project Directory**
```bash
mkdir deepseek-rag
cd deepseek-rag
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure

```
deepseek-rag/
├── main.py              # Core logic and utilities
├── api.py               # FastAPI application
├── requirements.txt     # Python dependencies
└── static/
    └── index.html       # Web interface
```

## Configuration

### 1. Environment Variables (Optional)

Create a `.env` file:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here  # Optional for local
```

### 2. Web Interface Configuration

After starting the server, configure through the web UI:

1. **Qdrant Setup**
   - URL: `http://localhost:6333` (for local) or your cloud URL
   - API Key: Leave empty for local, or enter your cloud API key

2. **RAG Mode**
   - Toggle RAG mode on/off
   - Adjust similarity threshold (0.0-1.0)

3. **Web Search**
   - Enable/disable web search fallback
   - Configure custom domains to search

## Running the Application

### Start the Server

```bash
python api.py
```

Or with uvicorn:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

## Usage Guide

### 1. Basic Chat (No RAG)

- Disable RAG mode in the sidebar
- Ask any question
- The model will respond using its knowledge

### 2. RAG Mode with Documents

1. **Enable RAG Mode**
   - Toggle "Enable RAG" in sidebar
   - Configure Qdrant connection

2. **Upload Documents**
   - **PDF Files**: Click "Upload PDF" and select file
   - **Web URLs**: Enter URL and click "Add URL"

3. **Ask Questions**
   - Questions will be answered using document context
   - See source documents in the response

### 3. Web Search Mode

1. Enable "Web Search" in sidebar
2. Configure search domains
3. Click 🌐 button to force web search for a query
4. Web search activates automatically if no relevant documents found

### 4. Advanced Features

**Thinking Process Visualization**
- Click "Show Thinking Process" in responses
- See the model's reasoning steps

**Source Attribution**
- Click "Show Sources" to see document references
- Each source shows excerpt and metadata

**Similarity Threshold**
- Adjust to control document relevance
- Higher = stricter matching
- Lower = more documents returned

## API Documentation

Once running, visit:
```
http://localhost:8000/docs
```

### Key Endpoints

#### Chat
```bash
POST /api/chat
{
  "message": "Your question here",
  "user_id": "user_123",
  "force_web_search": false,
  "rag_enabled": true
}
```

#### Upload PDF
```bash
POST /api/upload/pdf
Content-Type: multipart/form-data
file: <pdf_file>
```

#### Upload URL
```bash
POST /api/upload/url
Content-Type: application/x-www-form-urlencoded
url=https://example.com/article
```

#### Configure System
```bash
POST /api/configure
{
  "qdrant_url": "http://localhost:6333",
  "qdrant_api_key": "optional",
  "rag_enabled": true,
  "use_web_search": true,
  "similarity_threshold": 0.7,
  "search_domains": ["arxiv.org", "wikipedia.org"]
}
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# macOS/Linux
systemctl restart ollama

# Windows
# Restart from system tray
```

### Qdrant Connection Issues

```bash
# Check local Qdrant
curl http://localhost:6333/health

# Restart Qdrant Docker
docker restart <qdrant_container_id>
```

### Model Not Found

```bash
# Verify models are pulled
ollama list

# Pull missing models
ollama pull deepseek-r1:7b
ollama pull gpt-oss:20b
ollama pull snowflake-arctic-embed
```

### Memory Issues

If running out of memory:
1. Close other applications
2. Restart Ollama
3. Consider using lighter models
4. Increase Docker memory limit

## Performance Tips

1. **Model Warm-up**: First query may be slow
2. **Batch Processing**: Upload multiple documents before querying
3. **Threshold Tuning**: Adjust similarity threshold for better results
4. **Domain Filtering**: Limit web search domains for faster results

## Architecture

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   FastAPI   │
│   (api.py)  │
└──────┬──────┘
       │
       ├───→ ┌──────────────┐
       │     │   Ollama     │
       │     │ DeepSeek-R1  │
       │     └──────────────┘
       │
       ├───→ ┌──────────────┐
       │     │   Qdrant     │
       │     │ Vector Store │
       │     └──────────────┘
       │
       └───→ ┌──────────────┐
             │  Web Search  │
             │  GPT-OSS:20b │
             └──────────────┘
```

## Security Notes

- Use environment variables for sensitive data
- Don't expose Qdrant API keys in frontend
- Use HTTPS in production
- Implement rate limiting for production use

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t deepseek-rag .
docker run -p 8000:8000 deepseek-rag
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - feel free to use in your projects

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Check Ollama logs: `ollama logs`
- Check application logs in terminal

## Acknowledgments

- DeepSeek AI for the R1 model
- Ollama team for model serving
- Qdrant for vector database
- FastAPI for the web framework
