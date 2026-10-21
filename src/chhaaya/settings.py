from pydantic import Field
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
    gemini_api_key: str = Field(min_length=1)
    # Read by Compose for the optional `tunnel` profile, not by the service.
    cloudflare_tunnel_token: str | None = None
