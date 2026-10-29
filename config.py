import os
from pathlib import Path
# 数据目录可配置
BASE_DIR = Path(__file__).resolve().parent

data_dir = Path(os.getenv("TICKETFLOW_DATA_DIR", str(BASE_DIR)))
if not data_dir.is_absolute():
    data_dir = BASE_DIR / data_dir

DATA_DIR = data_dir.resolve()
#模型名称、timeout 和 max_tokens
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
MODEL_TIMEOUT_SECONDS = float(
    os.getenv("MODEL_TIMEOUT_SECONDS", "60")
)
MODEL_MAX_TOKENS = int(
    os.getenv("MODEL_MAX_TOKENS", "1200")
)

if MODEL_TIMEOUT_SECONDS <= 0:
    raise ValueError("MODEL_TIMEOUT_SECONDS 必须大于 0")

if MODEL_MAX_TOKENS <= 0:
    raise ValueError("MODEL_MAX_TOKENS 必须大于 0")