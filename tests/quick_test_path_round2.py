"""quick_test_path_round2 — PathExecutorRound2 単体テスト (LLM 必要)。

PathExecutorRound2 を 1-2 ペアで実行して Round 2 の改訂が動作することを
確認する。

検証項目:
- run() が PathResult を返す (round=2)
- revision_notes が空でない
- referenced_siblings が valid な PerspectiveID のみで構成 (空でも可)
- section_markdown が max_length=1700 内
- Tavily が呼ばれた場合は最大 MAX_ITERATIONS_ROUND2 (=2) 回まで

最小コストで動かすため、ハードコードされた own_draft + sibling_drafts を使う
(短めの mock section)。1 ケース ≈ ~$0.05-0.10。

実行例:
- python -m tests.quick_test_path_round2 --llm anthropic
- python -m tests.quick_test_path_round2 --llm anthropic --case-id 1
"""

import traceback
from datetime import datetime
from typing import Any, get_args

from langchain_core.language_models.chat_models import BaseChatModel

from src.main import (
    MAX_ITERATIONS_ROUND2,
    PathExecutorRound2,
)
from src.output_saver import save_output
from src.perspectives import (
    PERSPECTIVE_POOL,
    PathResult,
    PathSource,
    Perspective,
    PerspectiveID,
)


# ===== Mock drafts =====
# Round 1 から渡される既存 drafts を mock として hardcode。
# 実際の本番フローでは PathExecutor (Round 1) が生成する。

MOCK_AI_TECH_DRAFT = (
    "## AI/ML 技術観点: GraphCast 以降の系譜\n\n"
    "GraphCast (DeepMind 2023) は GNN ベースの全球気象予報モデルで、"
    "ERA5 を学習データに約 10 日間予報を生成する。"
    "従来の物理ベース NWP に対し計算コストを大幅削減しつつ精度面でも IFS を上回った。\n\n"
    "### ECMWF AIFS シリーズ\n"
    "AIFS は ECMWF が開発する Transformer ベースモデル。"
    "AIFS-ENS はアンサンブル版で、2025 年に実運用化した。\n\n"
    "### NOAA GraphCastGFS\n"
    "NOAA の GraphCast を GFS 初期値で動かす研究運用版。"
    "AWS Public Data で配信されている。"
)

MOCK_DATA_INFRA_DRAFT = (
    "## データ・観測インフラ観点: 学習データと配信基盤\n\n"
    "**ERA5** (ECMWF 全球再解析) は気象 AI モデルの事実上の標準学習データ。"
    "1979 年から現在までを ~31km/137 層でカバー。"
    "WeatherBench2 はこれを評価基盤として標準化した。\n\n"
    "### 配信フォーマット\n"
    "GRIB2 / NetCDF / Zarr が並立する。Python では xarray + cfgrib が解析標準。"
    "Herbie ライブラリは GFS / AIFS / GraphCastGFS を統一インタフェースで取得できる。\n\n"
    "### オープンデータ流通\n"
    "ECMWF Open Data (AIFS)、AWS Public Data Registry (NOAA GraphCastGFS)、"
    "Google Cloud (WeatherNext) などが並立。"
)

MOCK_GENAI_APPLICATION_DRAFT = (
    "## 生成 AI 活用観点: 気象解析エージェント\n\n"
    "気象 AI 研究では LLM が研究加速・コード生成・データ解析自動化に活用される。"
    "LangGraph は状態を持つエージェントフローを構築するフレームワークで、"
    "気象 AI 解析エージェントの実装に適する。\n\n"
    "### Microsoft Aurora\n"
    "Aurora は基盤モデル型の気象 AI で、複数下流タスクに転移できる。"
    "従来の専用モデル路線とは異なる位置づけ。\n\n"
    "### Anthropic / OpenAI\n"
    "Claude / GPT は気象データ解析の自然言語インタフェースとして使われる。"
)

MOCK_COMPARATIVE_INTL_DRAFT = (
    "## 国際比較観点: 主要機関の動向\n\n"
    "### ECMWF\n"
    "AIFS / AIFS-ENS / Anemoi で欧州協調体制の中心。Open Data 方針が積極的。\n\n"
    "### NOAA / NCEP\n"
    "GraphCastGFS で米国実験運用。AWS Open Data で配信。\n\n"
    "### Google DeepMind\n"
    "GraphCast / GenCast / WeatherNext シリーズの発信元。"
    "民間プレーヤーとして主導的位置づけ。"
)


# ===== テストケース =====
# 各ケースは (own_perspective, own_draft, sibling_pairs) のセット。
# sibling_pairs は (perspective_id, draft_markdown) のリスト。


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "label": "ai_tech が genai_application / data_infra / comparative_intl と peer-review",
        "query": (
            "GraphCast 登場以降の AI 気象予報モデルの技術系譜と、"
            "主要機関の動向を整理してください。"
        ),
        "own_perspective_id": "ai_tech",
        "own_draft": MOCK_AI_TECH_DRAFT,
        "sibling_pairs": [
            ("data_infra", MOCK_DATA_INFRA_DRAFT),
            ("genai_application", MOCK_GENAI_APPLICATION_DRAFT),
            ("comparative_intl", MOCK_COMPARATIVE_INTL_DRAFT),
        ],
    },
]


def _build_path_result(
    perspective_id: str, section_markdown: str
) -> PathResult:
    """ハードコードされた markdown から Round 1 PathResult を組み立てる。"""
    perspective = next(
        (p for p in PERSPECTIVE_POOL if p.id == perspective_id), None
    )
    if perspective is None:
        raise ValueError(f"Unknown perspective_id: {perspective_id}")
    return PathResult(
        perspective_id=perspective_id,  # type: ignore[arg-type]
        perspective_label=perspective.label,
        section_markdown=section_markdown,
        sources=[],  # mock では sources 空
        round=1,
    )


def run_test_for_case(
    executor: PathExecutorRound2,
    test_case: dict[str, Any],
    total: int,
) -> dict[str, Any]:
    """1 テストケースを実行し、結果と評価を返す。失敗時もエントリを返す。"""
    cid = test_case["id"]
    label = test_case["label"]
    query = test_case["query"]
    own_pid = test_case["own_perspective_id"]
    own_draft_md = test_case["own_draft"]
    sibling_pairs = test_case["sibling_pairs"]

    perspective = next(
        (p for p in PERSPECTIVE_POOL if p.id == own_pid), None
    )
    if perspective is None:
        return {
            "id": cid,
            "label": label,
            "success": False,
            "error": f"Unknown own_perspective_id: {own_pid}",
        }

    try:
        own_draft = _build_path_result(own_pid, own_draft_md)
        sibling_drafts = [
            _build_path_result(pid, md) for pid, md in sibling_pairs
        ]
    except Exception as e:
        return {
            "id": cid,
            "label": label,
            "success": False,
            "error": f"Mock build failed: {type(e).__name__}: {e}",
        }

    print(f"\n[Case {cid}/{total}] {label}")
    print(f"           query:           {query}")
    print(f"           own perspective: {own_pid} ({perspective.label})")
    print(
        f"           sibling count:   {len(sibling_drafts)} "
        f"({[s.perspective_id for s in sibling_drafts]})"
    )
    print(f"           own draft len:   {len(own_draft_md)} chars")
    print(
        f"           Running PathExecutorRound2 (max_iter={executor.max_iterations})..."
    )

    try:
        result: PathResult = executor.run(
            query=query,
            perspective=perspective,
            own_draft=own_draft,
            sibling_drafts=sibling_drafts,
        )
    except Exception as e:
        print(f"           [ERROR] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return {
            "id": cid,
            "label": label,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }

    section_len = len(result.section_markdown)
    sources_count = len(result.sources)
    valid_ids = set(get_args(PerspectiveID))
    referenced_siblings_valid = all(
        rid in valid_ids for rid in result.referenced_siblings
    )
    revision_notes_nonempty = bool(result.revision_notes.strip())
    section_within_max = section_len <= 1700
    round_is_2 = result.round == 2

    all_pass = (
        round_is_2
        and revision_notes_nonempty
        and referenced_siblings_valid
        and section_within_max
    )

    print(
        f"           section_len:     {section_len} chars "
        f"({'✓' if section_within_max else '✗'} <= 1700)"
    )
    print(
        f"           round:           {result.round} "
        f"({'✓' if round_is_2 else '✗'} expected 2)"
    )
    print(
        f"           refs:            {result.referenced_siblings} "
        f"({'✓' if referenced_siblings_valid else '✗'} valid PerspectiveIDs)"
    )
    print(
        f"           revision_notes:  '{result.revision_notes[:120]}'"
        f"{'...' if len(result.revision_notes) > 120 else ''}"
    )
    print(
        f"           notes_nonempty:  "
        f"{'✓' if revision_notes_nonempty else '✗'}"
    )
    print(f"           sources:         {sources_count}")
    for s in result.sources:
        print(f"             - {s.title[:60]}")
        print(f"               {s.url}")
    preview = result.section_markdown[:120].replace("\n", " ")
    print(f"           preview:         {preview}...")

    return {
        "id": cid,
        "label": label,
        "query": query,
        "own_perspective_id": own_pid,
        "perspective": perspective,
        "own_draft": own_draft,
        "sibling_drafts": sibling_drafts,
        "result": result,
        "section_len": section_len,
        "sources_count": sources_count,
        "round_is_2": round_is_2,
        "revision_notes_nonempty": revision_notes_nonempty,
        "referenced_siblings_valid": referenced_siblings_valid,
        "section_within_max": section_within_max,
        "all_pass": all_pass,
        "success": True,
        "error": None,
    }


def format_results_for_save(
    results: list[dict[str, Any]],
    llm_name: str,
    model_name: str,
) -> dict[str, str]:
    """結果を save_output 用の sections dict に変換。"""
    sections: dict[str, str] = {}

    sections["実行情報"] = (
        f"- LLM: {llm_name} / {model_name}\n"
        f"- Max iterations Round 2: {MAX_ITERATIONS_ROUND2}\n"
        f"- Tested cases: {len(results)}\n"
        f"- Run timestamp: {datetime.now().isoformat()}"
    )

    for r in results:
        title = f"Case {r['id']}: {r['label']}"
        if not r["success"]:
            sections[title] = (
                f"**ERROR**: {r['error']}\n\n"
                f"**Own perspective**: `{r.get('own_perspective_id', '?')}`"
            )
            continue

        result: PathResult = r["result"]
        siblings: list[PathResult] = r["sibling_drafts"]
        body_lines = [
            f"**Query**: {r['query']}",
            f"**Own perspective**: `{r['perspective'].id}` — {r['perspective'].label}",
            f"**Siblings**: {[s.perspective_id for s in siblings]}",
            f"**Own draft length**: {len(r['own_draft'].section_markdown)} chars",
            f"**Round 2 section length**: {r['section_len']} chars (max 1700)",
            f"**Round**: {result.round}",
            f"**Referenced siblings**: {result.referenced_siblings}",
            f"**Revision notes**: {result.revision_notes}",
            f"**Sources count**: {r['sources_count']}",
            "",
            "### Round 2 Section",
            "",
            result.section_markdown,
            "",
            "### Sources",
            "",
        ]
        if result.sources:
            for s in result.sources:
                body_lines.append(f"- [{s.title}]({s.url})")
                if s.snippet:
                    body_lines.append(f"  - {s.snippet[:200]}")
        else:
            body_lines.append("(なし)")
        sections[title] = "\n".join(body_lines)

    successful = [r for r in results if r["success"]]
    if successful:
        all_pass_count = sum(1 for r in successful if r["all_pass"])
        summary_lines = [
            f"- Successful runs: {len(successful)} / {len(results)}",
            f"- All checks passed: {all_pass_count} / {len(successful)}",
        ]
        sections["サマリ"] = "\n".join(summary_lines)
    else:
        sections["サマリ"] = "全テストケースで失敗しました。"

    return sections


def main():
    import argparse

    from src.settings import Settings, get_llm

    parser = argparse.ArgumentParser(
        description="quick_test_path_round2 — PathExecutorRound2 の単独動作確認"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="anthropic",
        choices=["openai", "anthropic"],
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=None,
        help="単一テストケースのみ実行 (1 から始まる ID)。指定なしで全ケース実行。",
    )
    args = parser.parse_args()

    settings = Settings()
    llm = get_llm(provider=args.llm, settings=settings)
    model_name = (
        settings.anthropic_smart_model
        if args.llm == "anthropic"
        else settings.openai_smart_model
    )

    if args.case_id is not None:
        cases_to_run = [c for c in TEST_CASES if c["id"] == args.case_id]
        if not cases_to_run:
            print(f"[ERROR] case-id {args.case_id} not found.")
            print(f"        Available IDs: {[c['id'] for c in TEST_CASES]}")
            return
        mode_label = f"SINGLE CASE (#{args.case_id})"
    else:
        cases_to_run = TEST_CASES
        mode_label = f"ALL {len(TEST_CASES)} CASES"

    total = len(cases_to_run)
    n_cases = total
    cost_min = n_cases * 0.05
    cost_max = n_cases * 0.15

    print(f"=== Run Configuration (quick_test_path_round2) ===")
    print(f"  Mode:           {mode_label}")
    print(f"  LLM:            {args.llm} / {model_name}")
    print(f"  Max iter R2:    {MAX_ITERATIONS_ROUND2}")
    print(f"  Section target: 1,200-1,400 chars (max 1700)")
    print(f"  Light LLM:      timeout=120s")
    print(f"  Expected cost:  ~${cost_min:.2f}-${cost_max:.2f} ({n_cases} cases)")
    print(f"  Expected time:  ~{n_cases * 30}-{n_cases * 90} seconds")
    print(f"  Stop anytime:   Ctrl+C で中断可能")
    print()

    executor = PathExecutorRound2(llm=llm)
    results = [run_test_for_case(executor, c, total) for c in cases_to_run]

    successful = [r for r in results if r["success"]]
    print(f"\n=== Summary ===")
    print(f"  Successful: {len(successful)} / {len(results)}")
    if successful:
        all_pass = sum(1 for r in successful if r["all_pass"])
        print(f"  All checks passed: {all_pass} / {len(successful)}")

    sections = format_results_for_save(results, args.llm, model_name)
    suffix = (
        f"{args.llm}_c{args.case_id}"
        if args.case_id is not None
        else f"{args.llm}_all"
    )
    saved_path = save_output(
        llm_name=f"path_round2_{suffix}",
        model_name=model_name,
        task=f"quick_test_path_round2 ({mode_label})",
        sections=sections,
    )
    print(f"\n[Saved] {saved_path}")


if __name__ == "__main__":
    main()
