"""通过命令行创建内部登录账号。"""

import argparse
import getpass
import os

from utils.auth_utils import get_auth_util


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 RAG 内部用户")
    parser.add_argument("username", help="登录用户名")
    parser.add_argument(
        "--password-env",
        help="从指定环境变量读取密码；不传时交互输入",
    )
    args = parser.parse_args()

    if args.password_env:
        password = os.getenv(args.password_env, "")
        if not password:
            raise RuntimeError(f"环境变量 {args.password_env} 未设置")
    else:
        password = getpass.getpass("密码: ")
        confirmation = getpass.getpass("再次输入密码: ")
        if password != confirmation:
            raise ValueError("两次输入的密码不一致")

    user = get_auth_util().create_user(
        username=args.username,
        password=password,
    )
    print(f"用户创建成功: {user['username']}")


if __name__ == "__main__":
    main()
