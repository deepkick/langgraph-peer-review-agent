"""quick_test_send_round2 — メイングラフ配線確認 (LLM 不要)。

メイングラフを Round 1 → round2_router → Round 2 → synthesizer の構造で
検証する。各 path executor は sleep スタブで模擬し、ほぼゼロコストで以下を
確認する:

検証項目:
1. Round 1 の N 並列が動く
2. round2_router の conditional_edges が機能する
3. peer_review_enabled=True で Round 2 の N 並列が動く
4. peer_review_enabled=False で Round 2 を skip して synthesizer に直行
5. wall time が (R1 max) + (R2 max) 程度 (≠ sum、並列性)
6. path_results_revised reducer (Annotated[list, operator.add]) が正しく動く

実行例:
- python -m tests.quick_test_send_round2 --n 4
- python -m tests.quick_test_send_round2 --n 2 --no-peer-review
- python -m tests.quick_test_send_round2 --seed 42
"""

import operator
import random
import re
import time
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.perspectives import (
    NUM_PATHS,
    PERSPECTIVE_POOL,
    PathResult,
    Perspective,
)


# ===== State 定義 (本番の MultiPathPeerReviewState を簡略化) =====


class _OverallState(BaseModel):
    """quick_test 用の簡略 State。

    path_results / path_results_revised は Annotated reducer で並列収集。
    """

    query: str
    perspectives: list[Perspective]
    path_results: Annotated[list[PathResult], operator.add] = Field(default_factory=list)
    path_results_revised: Annotated[list[PathResult], operator.add] = Field(
        default_factory=list
    )
    peer_review_enabled: bool = True


# ===== ノード関数 =====


def _selector_node(state: _OverallState) -> dict:
    """テスト用 selector: 既に確定済みの perspectives を表示するだけ。"""
    print(f"  [selector] using {len(state.perspectives)} perspectives:")
    for p in state.perspectives:
        print(f"    - {p.id} ({p.label})")
    return {}


def _fan_out_to_paths(state: _OverallState) -> list[Send]:
    """Round 1 への N 並列 fan-out。"""
    print(f"  [fan-out:R1] dispatching {len(state.perspectives)} parallel paths")
    return [
        Send(
            "path_executor",
            {
                "perspective_id": p.id,
                "perspective_label": p.label,
                "query": state.query,
            },
        )
        for p in state.perspectives
    ]


def _path_executor_node(input_state: dict) -> dict:
    """模擬 Round 1 path_executor: random sleep してから PathResult を返す。"""
    pid = input_state["perspective_id"]
    label = input_state["perspective_label"]
    delay = random.uniform(3.0, 7.0)

    start = time.time()
    print(f"    [path:{pid:20s}] R1 start (sleep {delay:.1f}s)")
    time.sleep(delay)
    elapsed = time.time() - start
    print(f"    [path:{pid:20s}] R1 done in {elapsed:.1f}s")

    return {
        "path_results": [
            PathResult(
                perspective_id=pid,
                perspective_label=label,
                section_markdown=f"## R1 Mock for {pid}\n\nSimulated R1 delay: {delay:.2f}s",
                sources=[],
                round=1,
            )
        ]
    }


def _round2_router_node(state: _OverallState) -> dict:
    """No-op gate.  Round 1 fan-in 後の conditional_edges を hang する空ノード。"""
    return {}


def _decide_round2(state: _OverallState) -> list[Send] | str:
    """Round 2 を回すか synthesizer に直行するか判定。"""
    if not state.peer_review_enabled:
        print(f"  [round2_router] peer_review disabled, skip Round 2")
        return "synthesizer"

    if len(state.path_results) != len(state.perspectives):
        print(
            f"  [round2_router] partial Round 1: "
            f"{len(state.path_results)}/{len(state.perspectives)}, skip Round 2"
        )
        return "synthesizer"

    print(f"  [round2_router] dispatching {len(state.path_results)} parallel R2 paths")
    sends = []
    for own in state.path_results:
        siblings = [
            other for other in state.path_results
            if other.perspective_id != own.perspective_id
        ]
        sends.append(
            Send(
                "path_executor_round2",
                {
                    "perspective_id": own.perspective_id,
                    "perspective_label": own.perspective_label,
                    "query": state.query,
                    "own_draft": own.model_dump(),
                    "sibling_drafts": [s.model_dump() for s in siblings],
                },
            )
        )
    return sends


def _path_executor_round2_node(input_state: dict) -> dict:
    """模擬 Round 2 path_executor: random sleep してから revised PathResult を返す。"""
    pid = input_state["perspective_id"]
    label = input_state["perspective_label"]
    delay = random.uniform(2.0, 5.0)  # Round 2 は少し短めに

    start = time.time()
    print(f"    [path:{pid:20s}] R2 start (sleep {delay:.1f}s)")
    time.sleep(delay)
    elapsed = time.time() - start
    print(f"    [path:{pid:20s}] R2 done in {elapsed:.1f}s")

    sibling_count = len(input_state.get("sibling_drafts", []))
    return {
        "path_results_revised": [
            PathResult(
                perspective_id=pid,
                perspective_label=label,
                section_markdown=f"## R2 Mock for {pid}\n\nSimulated R2 delay: {delay:.2f}s",
                sources=[],
                round=2,
                referenced_siblings=[],
                revision_notes=(
                    f"mock revision based on {sibling_count} siblings, "
                    f"delay {delay:.2f}s"
                ),
            )
        ]
    }


def _synthesizer_node(state: _OverallState) -> dict:
    """テスト用 synthesizer: どちらの結果セットを使うかを表示する。"""
    if (
        state.peer_review_enabled
        and len(state.path_results_revised) == len(state.perspectives)
    ):
        inputs = state.path_results_revised
        which = "Round 2 (revised)"
    else:
        inputs = state.path_results
        which = "Round 1 (fallback)"
    print(f"  [synthesizer] received {len(inputs)} results from {which}:")
    for r in inputs:
        m = re.search(r"Simulated R\d delay: ([\d.]+)s", r.section_markdown)
        delay_str = f" (delay {m.group(1)}s)" if m else ""
        print(f"    - {r.perspective_id} round={r.round}{delay_str}")
    return {}


# ===== Graph 構築 =====


def _build_graph():
    """本番グラフと同じトポロジを構築する (sleep スタブ版)。

    selector → (fan-out R1) → path_executor × N
    → round2_router → (fan-out R2 or skip)
      ├ path_executor_round2 × N → synthesizer
      └ synthesizer (skip)
    """
    graph = StateGraph(_OverallState)
    graph.add_node("selector", _selector_node)
    graph.add_node("path_executor", _path_executor_node)
    graph.add_node("round2_router", _round2_router_node)
    graph.add_node("path_executor_round2", _path_executor_round2_node)
    graph.add_node("synthesizer", _synthesizer_node)

    graph.add_edge(START, "selector")
    graph.add_conditional_edges(
        "selector",
        _fan_out_to_paths,
        ["path_executor"],
    )
    graph.add_edge("path_executor", "round2_router")
    graph.add_conditional_edges(
        "round2_router",
        _decide_round2,
        ["path_executor_round2", "synthesizer"],
    )
    graph.add_edge("path_executor_round2", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ===== テスト実行 =====


def run_test(
    perspective_ids: list[str], seed: int, peer_review_enabled: bool
) -> dict:
    """指定された perspective_ids で並列実行テストを行う。"""
    random.seed(seed)

    perspectives = []
    for pid in perspective_ids:
        p = next((x for x in PERSPECTIVE_POOL if x.id == pid), None)
        if p is None:
            raise ValueError(f"Unknown perspective_id: {pid}")
        perspectives.append(p)

    initial_state = _OverallState(
        query="GraphCast 以降の AI 気象予報モデルの技術系譜",
        perspectives=perspectives,
        path_results=[],
        path_results_revised=[],
        peer_review_enabled=peer_review_enabled,
    )

    print()
    print(f"  Initial state:")
    print(f"    query:               {initial_state.query}")
    print(f"    perspectives:        {[p.id for p in perspectives]}")
    print(f"    peer_review_enabled: {peer_review_enabled}")
    print()
    print(f"  Running graph...")
    print()

    graph = _build_graph()
    start = time.time()
    final_state = graph.invoke(initial_state)
    wall_time = time.time() - start

    # delays を抽出 (R1, R2 別々に)
    r1_delays = []
    for r in final_state["path_results"]:
        m = re.search(r"Simulated R1 delay: ([\d.]+)s", r.section_markdown)
        if m:
            r1_delays.append(float(m.group(1)))
    r2_delays = []
    for r in final_state.get("path_results_revised", []):
        m = re.search(r"Simulated R2 delay: ([\d.]+)s", r.section_markdown)
        if m:
            r2_delays.append(float(m.group(1)))

    n_paths = len(perspectives)

    return {
        "n_paths": n_paths,
        "peer_review_enabled": peer_review_enabled,
        "wall_time": wall_time,
        "r1_delays": r1_delays,
        "r2_delays": r2_delays,
        "r1_max": max(r1_delays) if r1_delays else 0,
        "r1_sum": sum(r1_delays),
        "r2_max": max(r2_delays) if r2_delays else 0,
        "r2_sum": sum(r2_delays),
        "path_results_count": len(final_state["path_results"]),
        "path_results_revised_count": len(
            final_state.get("path_results_revised", [])
        ),
    }


# ===== CLI エントリポイント =====


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "quick_test_send_round2 — メイングラフ配線確認 "
            "(Round 1 → round2_router → Round 2 → synthesizer)"
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=NUM_PATHS,
        help=f"並列パス数 (デフォルト: {NUM_PATHS}, 範囲: 1-8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random.seed 値 (再現性確保用、デフォルト: 42)",
    )
    parser.add_argument(
        "--no-peer-review",
        action="store_true",
        help="Round 2 を skip して synthesizer 直行する経路をテスト",
    )
    args = parser.parse_args()

    default_order = [
        "ai_tech",
        "data_infra",
        "genai_application",
        "comparative_intl",
        "meteorology",
        "disaster_society",
        "policy_ethics",
        "history_evolution",
    ]
    if args.n < 1 or args.n > len(default_order):
        print(f"[ERROR] --n must be in 1-{len(default_order)}")
        return
    selected_ids = default_order[: args.n]
    peer_review_enabled = not args.no_peer_review

    print(f"=== Run Configuration (quick_test_send_round2) ===")
    print(f"  Test:                graph wiring (R1 → router → R2 → synth)")
    print(f"  N paths:             {args.n}")
    print(f"  R1 mock delays:      random.uniform(3, 7) seconds per path")
    print(f"  R2 mock delays:      random.uniform(2, 5) seconds per path")
    print(f"  peer_review_enabled: {peer_review_enabled}")
    print(f"  Seed:                {args.seed}")
    print(f"  LLM calls:           0 (sleep-based simulation)")
    print(f"  Expected cost:       $0.00")
    if peer_review_enabled:
        print(f"  Expected time:       ~{args.n * 5 + args.n * 3.5}s sequential / ~12s parallel")
    else:
        print(f"  Expected time:       ~{args.n * 5}s sequential / ~7s parallel")

    result = run_test(
        selected_ids, seed=args.seed, peer_review_enabled=peer_review_enabled
    )

    print()
    print(f"=== Results ===")
    print(
        f"  Round 1 paths:        {result['path_results_count']} / "
        f"{result['n_paths']} expected "
        f"{'✓' if result['path_results_count'] == result['n_paths'] else '✗'}"
    )
    print(f"  R1 individual:        {[f'{d:.1f}s' for d in result['r1_delays']]}")
    print(f"  R1 sum:               {result['r1_sum']:.1f}s (sequential bound)")
    print(f"  R1 max:               {result['r1_max']:.1f}s (parallel bound)")

    if peer_review_enabled:
        expected_r2 = result["n_paths"]
        r2_count_ok = result["path_results_revised_count"] == expected_r2
        print(
            f"  Round 2 paths:        {result['path_results_revised_count']} / "
            f"{expected_r2} expected "
            f"{'✓' if r2_count_ok else '✗'}"
        )
        print(f"  R2 individual:        {[f'{d:.1f}s' for d in result['r2_delays']]}")
        print(f"  R2 sum:               {result['r2_sum']:.1f}s (sequential bound)")
        print(f"  R2 max:               {result['r2_max']:.1f}s (parallel bound)")
    else:
        if result["path_results_revised_count"] == 0:
            print(f"  Round 2 paths:        0 (skipped as expected) ✓")
        else:
            print(
                f"  Round 2 paths:        "
                f"{result['path_results_revised_count']} (UNEXPECTED with --no-peer-review) ✗"
            )

    print(f"  Actual wall time:     {result['wall_time']:.1f}s")
    print()

    print(f"=== Diagnosis ===")
    if peer_review_enabled:
        # 並列性: R1 max + R2 max ≈ wall time
        expected_parallel = result["r1_max"] + result["r2_max"]
        sequential_total = result["r1_sum"] + result["r2_sum"]
        is_parallel = result["wall_time"] < sequential_total * 0.7
        is_full_parallel = result["wall_time"] < expected_parallel * 1.3

        if is_full_parallel:
            print(
                f"  ✓ FULLY PARALLEL: wall ({result['wall_time']:.1f}s) "
                f"≈ R1max + R2max ({expected_parallel:.1f}s)"
            )
            print(f"    Both Round 1 and Round 2 dispatched in parallel.")
        elif is_parallel:
            print(
                f"  ✓ PARALLEL (with overhead): wall ({result['wall_time']:.1f}s) "
                f"< 0.7 × seq_total ({sequential_total * 0.7:.1f}s)"
            )
        else:
            print(
                f"  ✗ SEQUENTIAL?: wall ({result['wall_time']:.1f}s) "
                f">= 0.7 × seq_total ({sequential_total * 0.7:.1f}s)"
            )

        if result["path_results_revised_count"] == result["n_paths"]:
            print(
                f"  ✓ R2 REDUCER OK: collected {result['path_results_revised_count']}"
                f" results (Annotated[list, operator.add] on path_results_revised)"
            )
        else:
            print(
                f"  ✗ R2 REDUCER ISSUE: expected {result['n_paths']}"
                f" but got {result['path_results_revised_count']}"
            )
    else:
        # peer_review off の場合は Round 1 のみが並列実行される
        is_parallel = result["wall_time"] < result["r1_sum"] * 0.7
        is_full_parallel = result["wall_time"] < result["r1_max"] * 1.3
        if is_full_parallel:
            print(
                f"  ✓ R1 FULLY PARALLEL: wall ({result['wall_time']:.1f}s) "
                f"≈ R1max ({result['r1_max']:.1f}s), Round 2 skipped"
            )
        elif is_parallel:
            print(
                f"  ✓ R1 PARALLEL (with overhead): "
                f"wall ({result['wall_time']:.1f}s) < 0.7 × R1sum"
            )
        else:
            print(f"  ✗ SEQUENTIAL?: wall ({result['wall_time']:.1f}s)")

        if result["path_results_revised_count"] == 0:
            print(
                f"  ✓ ROUTER OK: Round 2 skipped as expected "
                f"(path_results_revised is empty)"
            )
        else:
            print(
                f"  ✗ ROUTER ISSUE: Round 2 was dispatched despite --no-peer-review"
            )


if __name__ == "__main__":
    main()
