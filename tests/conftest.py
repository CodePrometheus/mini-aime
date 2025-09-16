import os
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # 不覆盖已存在的环境变量（优先 IDE/CI 配置）
                os.environ.setdefault(key, val)
    except Exception:
        # 加载失败时静默，不影响测试运行
        pass


# 在测试会话开始前加载项目根目录的 .env
_load_dotenv()


