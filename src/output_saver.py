"""エージェント実行結果を Markdown ファイルとして保存するユーティリティ。"""

from datetime import datetime
from pathlib import Path

import yaml

# repo root: src/output_saver.py から見て parent.parent
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_output(
    llm_name: str,
    model_name: str,
    task: str,
    sections: dict[str, str],
) -> Path:
    """エージェントの実行結果を Markdown ファイルに保存する。

    保存先: <repo_root>/results/{YYYYMMDD_HHMMSS}_{llm_name}.md

    Args:
        llm_name: LLM プロバイダ名（"anthropic" / "openai"）
        model_name: 使用したモデル名（"claude-sonnet-4-6" 等）
        task: 入力タスク（CLI の --task 引数）
        sections: セクション名 → 内容 の辞書（dict の挿入順がセクション順になる）

    Returns:
        保存したファイルのパス。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filepath = RESULTS_DIR / f"{now.strftime('%Y%m%d_%H%M%S')}_{llm_name}.md"

    # YAML フロントマター
    metadata = {
        "pattern": "peer_review",
        "llm": llm_name,
        "model": model_name,
        "timestamp": now.isoformat(),
        "task": task,
    }
    yaml_str = yaml.dump(
        metadata,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    # 本文
    body = "\n".join(
        f"## {title}\n\n{content}\n" for title, content in sections.items()
    )

    filepath.write_text(f"---\n{yaml_str}---\n\n{body}", encoding="utf-8")
    return filepath
