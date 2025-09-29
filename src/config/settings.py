from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # LLM API 配置
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4000, alias="LLM_MAX_TOKENS")
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")

    # OpenRouter 兼容配置（可覆盖上面的通用配置）
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_api_base: str = Field(default="", alias="OPENROUTER_API_BASE")

    # 外部服务 API 配置
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    brave_search_api_key: str = Field(default="", alias="BRAVE_SEARCH_API_KEY")
    weather_api_key: str = Field(default="", alias="WEATHER_API_KEY")
    # 旅行相关服务（固定提供方）
    exchange_rate_api_key: str = Field(default="", alias="EXCHANGE_RATE_API_KEY")
    timezonedb_api_key: str = Field(default="", alias="TIMEZONEDB_API_KEY")
    nager_date_base_url: str = Field(default="https://date.nager.at", alias="NAGER_DATE_BASE_URL")
    nominatim_user_agent: str = Field(default="mini-aime/1.0 (contact@example.com)", alias="NOMINATIM_USER_AGENT")

    # 系统配置
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Metrics / Prometheus 配置
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=8001, alias="METRICS_PORT")

    # 人机环与工具重试配置
    human_in_loop_enabled: bool = Field(default=True, alias="HUMAN_IN_LOOP_ENABLED")
    default_tool_retry: int = Field(default=1, alias="DEFAULT_TOOL_RETRY")
    retry_backoff_ms: int = Field(default=800, alias="RETRY_BACKOFF_MS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
