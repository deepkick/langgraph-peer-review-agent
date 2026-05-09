"""Gradio app for multi-path peer-review agent demo (HF Spaces deployment).

Replay-only MVP: 録画済みの実行ログを時系列で再生する。
BYOK (Bring Your Own Key) は Phase 2.3 以降で追加予定。
"""
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

REPLAYS_DIR = Path(__file__).parent / "examples" / "replays"

# 認証 username は weather-ai-rag-hf-space と統一して "alumni"
STUDY_GROUP_USER = os.environ.get("STUDY_GROUP_USER", "alumni")
STUDY_GROUP_PASSWORD = os.environ.get("STUDY_GROUP_PASSWORD")


def _check_prerequisites() -> None:
    """起動前チェック: replay ファイル存在 + パスワード設定。

    weather-ai-rag-advanced の pattern を踏襲し、誤デプロイを防ぐため起動時に
    hard-fail させる。サイレントに「auth なしで起動」「replay 0 件で起動」を
    避けるのが目的。
    """
    if not REPLAYS_DIR.exists():
        print(f"ERROR: replays directory not found: {REPLAYS_DIR}", file=sys.stderr)
        sys.exit(1)

    json_files = list(REPLAYS_DIR.glob("*.json"))
    if not json_files:
        print(f"ERROR: no replay JSON files found in {REPLAYS_DIR}", file=sys.stderr)
        sys.exit(1)

    if not STUDY_GROUP_PASSWORD or STUDY_GROUP_PASSWORD == "your-shared-password-here":
        print(
            "ERROR: STUDY_GROUP_PASSWORD is not set in .env "
            "(or still holds the placeholder value).",
            file=sys.stderr,
        )
        sys.exit(1)


_check_prerequisites()


def load_all_replays() -> list[dict]:
    """examples/replays/*.json を全て読み込み、metadata でソートして返す。

    1 つでも malformed な JSON / 必要キー欠落の file があってもアプリ全体は
    起動する (該当 file はスキップして warning を出す)。replay が増えても
    1 件の事故で demo 全体が落ちない設計。
    """
    replays = []
    for json_file in sorted(REPLAYS_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            replays.append({
                "id": data["metadata"]["id"],
                "title": data["metadata"]["title"],
                "description": data["metadata"]["description"],
                "data": data,
            })
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[WARN] {json_file.name} のロードに失敗、スキップ: {e}")
    return replays


REPLAYS = load_all_replays()
REPLAY_BY_ID = {r["id"]: r for r in REPLAYS}
DROPDOWN_CHOICES = [(f"{r['title']} — {r['description']}", r["id"]) for r in REPLAYS]


def format_stage_log(stage: dict, elapsed: float) -> str:
    """1 stage の表示行を整形 (CLI ログ風)。"""
    s = stage["stage"]
    e = f"[{elapsed:6.1f}s]"
    if s == "selector_done":
        ids = [p["id"] for p in stage["perspectives"]]
        return f"{e} [selector] selected: {ids}"
    if s == "round1_path_done":
        return (f"{e}   [path:{stage['perspective_id']}] done: "
                f"{len(stage['section_markdown'])} chars, "
                f"{len(stage['sources'])} sources")
    if s == "round2_path_done":
        notes = stage["revision_notes"][:60]
        if len(stage["revision_notes"]) > 60:
            notes += "..."
        return (f"{e}   [path_round2:{stage['perspective_id']}] done: "
                f"{len(stage['section_markdown'])} chars, "
                f"refs={stage['referenced_siblings']}, "
                f"notes='{notes}'")
    if s == "synthesizer_done":
        return (f"{e} [synthesizer] done: "
                f"intro {len(stage['introduction'])} / "
                f"cross {len(stage['cross_path_observations'])} / "
                f"conclusion {len(stage['conclusion'])} chars")
    if s == "assembly_done":
        return f"{e} [assembly] final_report: {len(stage['final_report'])} chars"
    return f"{e} [{s}]"


def replay_generator(replay_id: str, speed: float):
    """選択された replay を再生する generator。

    Yields:
        (log_text, final_report_md): UI に流す累積ログと最終レポート
    """
    if replay_id is None or replay_id not in REPLAY_BY_ID:
        yield "(replay を選択してください)", ""
        return

    data = REPLAY_BY_ID[replay_id]["data"]
    stages = data["stages"]
    log_lines = []
    final_report = ""

    log_lines.append(f"=== Replay: {data['metadata']['title']} ===")
    log_lines.append(f"Task: {data['metadata']['task'][:80]}...")
    log_lines.append(f"Total elapsed (original): {data['metadata']['total_elapsed_seconds']:.0f}s")
    log_lines.append("")
    yield "\n".join(log_lines), ""

    prev_elapsed = 0.0
    for stage in stages:
        # 前 stage との elapsed 差を speed で割って sleep (UX cap で 5 秒上限)
        wait = (stage["elapsed_seconds"] - prev_elapsed) / max(speed, 1.0)
        time.sleep(min(wait, 5.0))
        prev_elapsed = stage["elapsed_seconds"]

        log_lines.append(format_stage_log(stage, stage["elapsed_seconds"]))

        if stage["stage"] == "assembly_done":
            final_report = stage["final_report"]

        yield "\n".join(log_lines), final_report


with gr.Blocks(
    title="Multi-Path Peer-Review Agent Demo",
) as demo:
    gr.Markdown("# Multi-Path Peer-Review Agent — 勉強会 Demo")
    gr.Markdown(
        "気象 × AI 領域の研究レポートを 4 視点並列で生成し、Round 2 で相互 peer-review する "
        "LangGraph エージェントの **replay デモ** です。"
        "実際の実行結果 (~4 分 / ~$1.5) を録画したものを時系列で再生します。"
    )

    with gr.Row():
        replay_select = gr.Dropdown(
            choices=DROPDOWN_CHOICES,
            label="Replay 選択",
            value=DROPDOWN_CHOICES[0][1] if DROPDOWN_CHOICES else None,
        )
    with gr.Row():
        speed = gr.Slider(1, 30, value=10, step=1, label="再生速度倍率 (10× 推奨)")
        play_btn = gr.Button("▶ 再生", variant="primary", scale=0)

    log_box = gr.Textbox(label="実行ログ (stream)", lines=18, max_lines=18)
    report_box = gr.Markdown(
        label="最終レポート",
        value="(再生完了後に表示されます)",
    )

    play_btn.click(
        fn=replay_generator,
        inputs=[replay_select, speed],
        outputs=[log_box, report_box],
    )

    gr.Markdown(
        "---\n\n"
        "**参考**: ソースコード / 設計詳細は "
        "[GitHub: deepkick/langgraph-peer-review-agent]"
        "(https://github.com/deepkick/langgraph-peer-review-agent) を参照。"
    )


if __name__ == "__main__":
    # weather-ai-rag-advanced と同様、queue + concurrency を制限
    # max_size=5: 待機キューを 5 までに抑制
    # default_concurrency_limit=1: 同時 1 リクエストのみ処理 (replay は軽いが念のため)
    demo.queue(max_size=5, default_concurrency_limit=1)
    # auth は _check_prerequisites() で password 確定済みのため fail-fast OK
    # theme は Gradio 6.0 で Blocks() → launch() に移動した API
    demo.launch(
        auth=(STUDY_GROUP_USER, STUDY_GROUP_PASSWORD),
        # システムフォント (SF Pro / Segoe UI / Hiragino Sans 等) に
        # フォールバックさせ、Latin と日本語の混在時の可読性を確保。
        # Soft theme のデフォルト (Quicksand) は丸みが強く、技術文書の
        # 可読性に欠けるため明示的に上書きする。
        # Gradio 6.x は font / font_mono に Font オブジェクトを期待するため
        # 生 str ではなく gr.themes.Font(name) で wrap する。
        theme=gr.themes.Soft(
            font=[
                gr.themes.Font("ui-sans-serif"),
                gr.themes.Font("system-ui"),
                gr.themes.Font("-apple-system"),
                gr.themes.Font("sans-serif"),
            ],
            font_mono=[
                gr.themes.Font("ui-monospace"),
                gr.themes.Font("SF Mono"),
                gr.themes.Font("Consolas"),
                gr.themes.Font("monospace"),
            ],
        ),
    )
