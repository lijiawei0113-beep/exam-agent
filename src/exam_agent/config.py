"""集中读取配置。所有可变参数都从环境变量 / .env 读取，代码里不写死密钥。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    thinking: str  # "enabled" | "disabled"
    db_path: Path
    blind_model: str = ""      # 盲解 Agent 单独使用的模型；空 = 与命题员相同
    blind_thinking: str = ""   # 盲解 Agent 是否开启思考模式；空 = 与命题员相同


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    db_path = Path(os.getenv("EXAM_DB_PATH", "data/exam.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        thinking=os.getenv("DEEPSEEK_THINKING", "disabled"),
        db_path=db_path,
        blind_model=os.getenv("EXAM_BLIND_MODEL", ""),
        blind_thinking=os.getenv("EXAM_BLIND_THINKING", ""),
    )
