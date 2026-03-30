"""
key_loader.py
统一加载 API Key，支持环境变量和项目根目录 API_Key.json。
"""

import json
import os
from typing import Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_CONFIG_PATH = os.path.join(PROJECT_ROOT, "API_Key.json")


def load_api_key_config() -> dict:
    """加载项目根目录 API_Key.json 配置。"""
    if not os.path.exists(API_KEY_CONFIG_PATH):
        return {}

    try:
        with open(API_KEY_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_api_key(provider: str) -> Optional[str]:
    """
    根据 provider 获取 API Key。

    provider:
        - deepseek
        - dashscope
    """
    provider = (provider or "").strip().lower()
    cfg = load_api_key_config()

    if provider == "deepseek":
        return (
            os.getenv("DEEPSEEK_API_KEY")
            or cfg.get("deepseek_api_key")
            or None
        )

    if provider == "dashscope":
        return (
            os.getenv("DASHSCOPE_API_KEY")
            or cfg.get("dashscope_api_key")
            or None
        )

    return None
