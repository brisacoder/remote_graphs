from typing import Any, Literal

from pydantic_settings import BaseSettings


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):

    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    PROJECT_NAME: str = "Remote Graphs Application"
    DESCRIPTION: str = "Application to demonstrate remote graphs"


settings = Settings()  # type: ignore
