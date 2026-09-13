"""MongoDB 用户认证、密码哈希和登录令牌工具。"""

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from bson import ObjectId
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from config.auth_config import AuthConfig, auth_config
from utils.mongodb_utils import get_mongodb_util


class AuthUtil:
    """管理内部账号并签发短期 JWT 登录凭证。"""

    def __init__(self, config: AuthConfig | None = None) -> None:
        self.config = config or auth_config
        self.mongodb = get_mongodb_util()
        self.collection = self.mongodb.database[
            self.mongodb.config.user_collection
        ]
        self.password_hash = PasswordHash.recommended()
        self.collection.create_index(
            [("username_normalized", ASCENDING)],
            name="username_normalized_unique_idx",
            unique=True,
        )

    @staticmethod
    def _normalize_username(username: str) -> tuple[str, str]:
        username = username.strip()
        if not 3 <= len(username) <= 64:
            raise ValueError("用户名长度必须在 3 到 64 个字符之间")
        if any(character.isspace() for character in username):
            raise ValueError("用户名不能包含空白字符")
        return username, username.casefold()

    @staticmethod
    def _validate_password(password: str) -> None:
        if not 5 <= len(password) <= 128:
            raise ValueError("密码长度必须在 5 到 128 个字符之间")

    def create_user(self, *, username: str, password: str) -> dict[str, Any]:
        username, normalized = self._normalize_username(username)
        self._validate_password(password)
        now = datetime.now(timezone.utc)
        try:
            result = self.collection.insert_one(
                {
                    "username": username,
                    "username_normalized": normalized,
                    "password_hash": self.password_hash.hash(password),
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        except DuplicateKeyError as exc:
            raise ValueError("用户名已经存在") from exc
        return self._public_user(
            self.collection.find_one({"_id": result.inserted_id})
        )

    def authenticate(self, *, username: str, password: str) -> dict[str, Any] | None:
        _, normalized = self._normalize_username(username)
        user = self.collection.find_one({"username_normalized": normalized})
        if not user or not user.get("is_active", False):
            return None
        if not self.password_hash.verify(password, user.get("password_hash", "")):
            return None
        return self._public_user(user)

    def create_access_token(self, user: dict[str, Any]) -> str:
        self.config.validate()
        now = datetime.now(timezone.utc)
        return jwt.encode(
            {
                "sub": user["id"],
                "username": user["username"],
                "iat": now,
                "exp": now + timedelta(hours=self.config.token_expire_hours),
            },
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def get_user_from_token(self, token: str) -> dict[str, Any] | None:
        self.config.validate()
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )
            user_id = payload.get("sub")
            if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
                return None
        except InvalidTokenError:
            return None

        user = self.collection.find_one(
            {"_id": ObjectId(user_id), "is_active": True}
        )
        return self._public_user(user) if user else None

    @staticmethod
    def _public_user(user: dict[str, Any] | None) -> dict[str, Any]:
        if not user:
            raise RuntimeError("用户记录不存在")
        return {
            "id": str(user["_id"]),
            "username": str(user["username"]),
            "created_at": user.get("created_at"),
        }


_auth_util: AuthUtil | None = None


def get_auth_util() -> AuthUtil:
    global _auth_util
    if _auth_util is None:
        _auth_util = AuthUtil()
    return _auth_util
