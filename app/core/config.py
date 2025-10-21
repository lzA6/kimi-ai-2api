from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra="ignore"
    )

    APP_NAME: str = "kimi-ai-2api"
    APP_VERSION: str = "1.0.0"
    DESCRIPTION: str = "一个将 kimi-ai.chat 转换为兼容 OpenAI 格式 API 的高性能代理。"

    API_MASTER_KEY: Optional[str] = None
    
    API_REQUEST_TIMEOUT: int = 180
    NGINX_PORT: int = 8088
    SESSION_CACHE_TTL: int = 3600

    KNOWN_MODELS: List[str] = ["kimi-k2-instruct-0905", "kimi-k2-instruct"]
    DEFAULT_MODEL: str = "kimi-k2-instruct-0905"
    
    UPSTREAM_URL: str = "https://kimi-ai.chat/wp-admin/admin-ajax.php"
    CHAT_PAGE_URL: str = "https://kimi-ai.chat/chat/"

settings = Settings()
