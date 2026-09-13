"""FastAPI 通用依赖。"""

from typing import Annotated, Any

from fastapi import Cookie, Depends, HTTPException, status

from config.auth_config import auth_config
from utils.auth_utils import get_auth_util


def get_current_user(
    token: Annotated[
        str | None,
        Cookie(alias=auth_config.cookie_name),
    ] = None,
) -> dict[str, Any]:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
        )
    user = get_auth_util().get_user_from_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
        )
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]
