"""登录认证配置。"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _as_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AuthConfig:
    secret_key: str = field(default_factory=lambda: os.getenv("AUTH_SECRET_KEY", ""))
    algorithm: str = "HS256"
    cookie_name: str = field(
        default_factory=lambda: os.getenv("AUTH_COOKIE_NAME", "rag_session")
    )
    token_expire_hours: int = field(
        default_factory=lambda: int(os.getenv("AUTH_TOKEN_EXPIRE_HOURS", "24"))
    )
    cookie_secure: bool = field(
        default_factory=lambda: _as_bool("AUTH_COOKIE_SECURE")
    )

    def validate(self) -> None:
        if len(self.secret_key) < 32:
            raise RuntimeError("AUTH_SECRET_KEY 至少需要 32 个字符")
        if self.token_expire_hours <= 0:
            raise RuntimeError("AUTH_TOKEN_EXPIRE_HOURS 必须大于 0")


auth_config = AuthConfig()
