import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_application_credentials: str = ""
    project_id: str = ""
    location: str = "us-central1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # ignore env vars not defined in the class

settings = Settings()

# Set GOOGLE_APPLICATION_CREDENTIALS in env so google client libraries pick it up
if settings.google_application_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
