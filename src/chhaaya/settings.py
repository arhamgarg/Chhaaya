from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="forbid", hide_input_in_errors=True
    )

    postgres_password: str
    database_url: str
    whatsapp_verify_token: str = Field(min_length=1)
    whatsapp_app_secret: str = Field(min_length=1)
    whatsapp_access_token: str = Field(min_length=1)
    whatsapp_phone_number_id: str = Field(min_length=1)
    ai_provider: Literal["claude", "sarvam"] = "claude"
    anthropic_api_key: str = ""
    sarvam_api_key: str = ""
    speech_enabled: bool = False
    intent_model_path: Path = Path("data/intent/model")
    # Read by Compose for the optional `tunnel` profile, not by the service.
    cloudflare_tunnel_token: str | None = None

    @model_validator(mode="after")
    def provider_credentials(self) -> Self:
        if self.ai_provider == "claude" and not self.anthropic_api_key.strip():
            raise ValueError("ANTHROPIC_API_KEY is required for AI_PROVIDER=claude")
        if (self.ai_provider == "sarvam" or self.speech_enabled) and not (
            self.sarvam_api_key.strip()
        ):
            raise ValueError("SARVAM_API_KEY is required for Sarvam or enabled speech")
        return self
