"""multi_path_peer_review の Pydantic データモデル定義。

State / Synthesizer 出力 / 各 PathExecutor の LLM 出力を集約する。
"""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from .perspectives import (
    PathResult,
    PathSource,
    Perspective,
    PerspectiveID,
)


# Round 2 (peer-review) の section_markdown 上限。
# Round 1 の max_length=1700 とは独立した定数として定義する。
# Round 2 は改訂で fact correction / cross-reference / gap filling を行うため、
# 構造的に文字数増が必要となる。Round 1 の strict gate (max_length=1700) が
# 下流のベースラインを保証するため、Round 2 の緩和は defense-in-depth を
# 破らない。
#
# 値の根拠: 2500 は「response 制御 (= prompt で Round 2 を最小限の差分と定義)
# で実出力を Round 1 ± 200 字程度に抑制し、2500 はあくまで異常時の構造的防御」
# という設計に基づく。2500 を超えるなら LLM の response 制御が機能していない
# 兆候であり、単に上限を引き上げるのではなく prompt の見直しが必要。
MAX_SECTION_MARKDOWN_LENGTH_ROUND2 = 2500


# ===== State 定義 =====


class MultiPathPeerReviewState(BaseModel):
    """multi_path_peer_review 全体 State。

    - path_results: Round 1 の出力 (Annotated reducer)
    - path_results_revised: Round 2 の出力 (Annotated reducer)
    - peer_review_enabled: A/B 比較用 toggle (CLI から制御)
    """

    query: str
    perspectives: list[Perspective] = Field(default_factory=list)
    selector_rationale: str = ""

    # Round 1 の出力
    path_results: Annotated[list[PathResult], operator.add] = Field(default_factory=list)

    # Round 2 の出力
    path_results_revised: Annotated[list[PathResult], operator.add] = Field(
        default_factory=list
    )

    # Round 2 を回すかの toggle (CLI から制御)
    peer_review_enabled: bool = True

    synthesizer_output: "SynthesizerOutput | None" = None
    final_report: str = ""


# ===== Synthesizer の出力型 =====


class SynthesizerOutput(BaseModel):
    """cross_path_synthesizer の Pydantic 出力。

    Heavy LLM の出力は intro / cross_path_observations / conclusion の 3 つだけ。
    本文 (各 PathResult.section_markdown) は rewrite せず、コード側で連結する。
    """

    introduction: str = Field(
        ...,
        max_length=600,
        description=(
            "ユーザー目標と選定された視点の構成を 200-400 文字で説明する導入。"
            "なぜこの視点群で目標を多角的にカバーできるかを示す。"
        ),
    )
    cross_path_observations: str = Field(
        ...,
        max_length=900,
        description=(
            "視点間の横断的観察を 300-500 文字で記述。multi_path 設計の核心。"
            "(1) 一致点 (複数視点が同じ結論を支持)、"
            "(2) 相補点 (視点が補完的に組み合わさる)、"
            "(3) 緊張点 (視点間で見方が分かれる) を扱う。"
            "視点 ID を明示しながら記述する。"
        ),
    )
    conclusion: str = Field(
        ...,
        max_length=500,
        description=(
            "全体結論と示唆を 200-300 文字で記述。"
            "視点群を経て見えてきた重要ポイントを凝縮する。"
        ),
    )


# Forward reference の解決
MultiPathPeerReviewState.model_rebuild()


# ===== PathExecutor (Round 1) の LLM 出力型 =====


class _PathLLMOutput(BaseModel):
    """PathExecutor の LLM 直接生成物 (内部用)。

    perspective_id / perspective_label は呼び出し側が注入するため除外。
    """

    section_markdown: str = Field(
        ...,
        max_length=1700,
        description=(
            "担当視点に基づくセクション本文。Markdown、見出し ## から開始。"
            "**1,200-1,400 文字以内厳守 (1,500 字超過禁止、上限 1700)**。"
            "視点の核心を凝縮し、サブセクションは 3 個以内、各 200-400 字程度。"
            "詳細は出典 URL に委ね、本文は要点のみ。"
        ),
    )
    sources: list[PathSource] = Field(
        default_factory=list,
        description="このパスで実際に section 内で引用した Tavily 結果。",
    )


# ===== PathExecutorRound2 の LLM 出力型 =====


class _PathRound2LLMOutput(BaseModel):
    """PathExecutorRound2 の LLM 直接生成物 (内部用)。

    Round 2 用フィールドとして revision_notes, referenced_siblings を持つ。
    perspective_id / perspective_label は呼び出し側で注入するため除外。
    """

    section_markdown: str = Field(
        ...,
        max_length=MAX_SECTION_MARKDOWN_LENGTH_ROUND2,
        description=(
            "改訂後の section 本文。Markdown、見出し ## から開始。"
            "Round 2 では Round 1 より文字数の柔軟性を許容: target 1,300-1,600 字、"
            "改訂目標達成のためなら 2,000 字程度まで許容、"
            f"上限 max_length={MAX_SECTION_MARKDOWN_LENGTH_ROUND2} 字。"
            "改訂不要の場合は Round 1 draft をそのまま再出力してよい。"
        ),
    )
    sources: list[PathSource] = Field(
        default_factory=list,
        description="改訂後の引用ソース一覧。Round 1 の sources をベースに、矛盾解消で追加した分のみ追加。",
    )
    revision_notes: str = Field(
        ...,
        max_length=400,
        description=(
            "何をどう変えたか / なぜ。1-3 文 (200 字以内目安)。"
            "改訂不要の場合は exactly 'significant change not required' と書く。"
        ),
    )
    referenced_siblings: list[PerspectiveID] = Field(
        default_factory=list,
        description=(
            "本文中で実際に参照した他視点の id のみ。形式的に全部入れない。"
            "参照しなかった場合は空リスト。"
        ),
    )
