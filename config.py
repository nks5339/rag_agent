"""
Configuration management for DeepSeek RAG Agent
"""
import os
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QdrantConfig(BaseModel):
    """Qdrant configuration"""
    url: str = Field(default="http://localhost:6333")
    api_key: Optional[str] = Field(default=None)
    timeout: int = Field(default=60)


class ModelConfig(BaseModel):
    """Model configuration"""
    main_model: str = Field(default="deepseek-r1:7b")
    web_search_model: str = Field(default="gpt-oss:20b")
    embedding_model: str = Field(default="snowflake-arctic-embed")
    embedding_dimensions: int = Field(default=1024)


class RAGConfig(BaseModel):
    """RAG configuration"""
    collection_name: str = Field(default="deepseek-r1-rag")
    similarity_threshold: float = Field(default=0.7)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_results: int = Field(default=5)


class WebSearchConfig(BaseModel):
    """Web search configuration"""
    enabled: bool = Field(default=False)
    default_domains: List[str] = Field(
        default=["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    )
    max_results: int = Field(default=5)


class AppConfig(BaseModel):
    """Main application configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "False").lower() == "true",
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=int(os.getenv("QDRANT_TIMEOUT", "60"))
            ),
            models=ModelConfig(
                main_model=os.getenv("MODEL_VERSION", "deepseek-r1:7b"),
                web_search_model=os.getenv("WEB_SEARCH_MODEL", "gpt-oss:20b"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed"),
                embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
            ),
            rag=RAGConfig(
                collection_name=os.getenv("COLLECTION_NAME", "deepseek-r1-rag"),
                similarity_threshold=float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7")),
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                max_results=int(os.getenv("MAX_RESULTS", "5"))
            ),
            web_search=WebSearchConfig(
                enabled=os.getenv("ENABLE_WEB_SEARCH", "False").lower() == "true",
                default_domains=os.getenv("SEARCH_DOMAINS", 
                    "arxiv.org,wikipedia.org,github.com,medium.com").split(","),
                max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
            )
        )
    
    def update_from_dict(self, updates: dict) -> None:
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # Update nested config
                    nested_config = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)


# Global configuration instance
config = AppConfig.from_env()


def get_config() -> AppConfig:
    """Get global configuration instance"""
    return config


def update_config(updates: dict) -> AppConfig:
    """Update global configuration"""
    config.update_from_dict(updates)
    return config


if __name__ == "__main__":
    # Test configuration
    print("Current Configuration:")
    print(f"Host: {config.host}:{config.port}")
    print(f"Debug: {config.debug}")
    print(f"Qdrant URL: {config.qdrant.url}")
    print(f"Main Model: {config.models.main_model}")
    print(f"Collection: {config.rag.collection_name}")
    print(f"Web Search: {config.web_search.enabled}")
