"""登录、退出和当前用户接口。"""

from fastapi import APIRouter, HTTPException, Response, status

from api.dependencies import CurrentUser
from api.schemas import LoginRequest, UserResponse
from config.auth_config import auth_config
from utils.auth_utils import get_auth_util

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserResponse, summary="账号密码登录")
def login(payload: LoginRequest, response: Response) -> UserResponse:
    auth = get_auth_util()
    try:
        user = auth.authenticate(
            username=payload.username,
            password=payload.password,
        )
    except ValueError:
        user = None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    token = auth.create_access_token(user)
    response.set_cookie(
        key=auth_config.cookie_name,
        value=token,
        max_age=auth_config.token_expire_hours * 3600,
        httponly=True,
        secure=auth_config.cookie_secure,
        samesite="lax",
        path="/",
    )
    return UserResponse.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="退出登录")
def logout(response: Response) -> None:
    response.delete_cookie(
        key=auth_config.cookie_name,
        path="/",
        secure=auth_config.cookie_secure,
        httponly=True,
        samesite="lax",
    )


@router.get("/me", response_model=UserResponse, summary="读取当前登录用户")
def me(current_user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(current_user)
