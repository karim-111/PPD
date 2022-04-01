from pydantic import BaseSettings


class Settings(BaseSettings):
    CONSUMER_KEY: str
    CONSUMER_SECRET: str
    ACCESS_KEY: str
    ACCESS_SECRET: str

    class Config:
        env_file = ".env"
