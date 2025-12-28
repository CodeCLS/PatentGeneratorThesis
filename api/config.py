"""
Configuration for the API.
"""
import os
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database settings
    database_type: Literal["local", "postgres", "supabase"] = "local"
    
    # PostgreSQL/Supabase settings (if using)
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_db: str = os.getenv("POSTGRES_DB", "patent_kg")
    
    # Supabase settings
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    
    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Authentication settings
    auth_secret: str = os.getenv("AUTH_SECRET", "")
    
    # Redis settings (optional, for resumable streams)
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # File storage settings
    storage_type: str = os.getenv("STORAGE_TYPE", "local")  # local, s3, azure, vercel
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "")
    storage_region: str = os.getenv("STORAGE_REGION", "")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # Rate limiting
    rate_limit_messages_guest: int = int(os.getenv("RATE_LIMIT_MESSAGES_GUEST", "50"))
    rate_limit_messages_regular: int = int(os.getenv("RATE_LIMIT_MESSAGES_REGULAR", "1000"))
    
    # File upload limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()




