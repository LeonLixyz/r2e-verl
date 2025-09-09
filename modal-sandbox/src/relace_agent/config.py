from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_file_as


class BaseConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class FileConfig(BaseConfig):
    @classmethod
    def load(cls, path: Path) -> Self:
        return parse_yaml_file_as(cls, path)


class AgentConfig(BaseConfig):
    system: str
    user: str
    tools: list[str]
    model_name: str = "anthropic/claude-sonnet-4"
    max_tokens: int = 8192
    max_turns: int = 200
    prompt_timeout: int = 180  # 3 minutes
    prompt_retries: int = 1


class BuildConfig(BaseConfig):
    install: list[str] = Field(default_factory=lambda: ["npm", "install"])
    build: list[str] = Field(default_factory=lambda: ["npm", "run", "build"])
    timeout: int | None = None


class TestConfig(BaseConfig):
    serve: list[str] = Field(default_factory=lambda: ["npm", "run", "preview"])
    serve_url: str = "http://localhost:4173"
    wait_load: int = 5_000
    wait_error: int = 5_000


class DeployConfig(BaseConfig):
    dist_path: str = "dist"


class RelaceConfig(FileConfig):
    user_agents: dict[str, AgentConfig]
    build_agent: AgentConfig
    test_agent: AgentConfig

    build: BuildConfig = Field(default_factory=BuildConfig)
    test: TestConfig = Field(default_factory=TestConfig)
    deploy: DeployConfig = Field(default_factory=DeployConfig)

    # TODO: Implement versioning

    @classmethod
    def load_default(cls) -> Self:
        return cls.load(Path(__file__).parent.parent.parent / "relace.yaml")
