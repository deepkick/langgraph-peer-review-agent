"""多経路 peer-review エージェント本体。

各視点が独立に Tavily 検索 + section 執筆 (Round 1) を行い、Round 2 で
他視点の draft を参照して改訂、Synthesizer で横断観察を加えて最終 Markdown
を組み立てる。

設計の概要:
- Round 1: 各視点が独立に Tavily 検索 + section 執筆
- Round 2 (peer-review): 各視点が他視点の draft を踏まえて自分の section を改訂
  - 矛盾解消用途のみ Tavily を最大 MAX_ITERATIONS_ROUND2 (=2) 回まで許可
  - 「改訂不要」を許容 (revision_notes に exactly
    'significant change not required' で表明)
  - referenced_siblings には本文中で実際に参照した他視点の id のみ
- Synthesizer: Round 2 が完走していれば revised を、そうでなければ Round 1 を入力
- Assembly: 同上

サブエージェント:
- PerspectiveSelector (Light): curated list から視点を選定
- PathExecutor (Light): Round 1 用 path executor
- PathExecutorRound2 (Light): 他視点 drafts を読んで section を改訂
- CrossPathSynthesizer (Heavy): 視点横断の観察を生成

State 流れ:
- query: ユーザー目標
- perspectives: selector が決定
- path_results: Round 1 結果 (Annotated reducer)
- path_results_revised: Round 2 結果 (Annotated reducer)
- peer_review_enabled: A/B 比較用 toggle (CLI から制御)
- synthesizer_output: Heavy LLM の生成物
- final_report: コード側で組み立てた最終 Markdown

実行例:
- デモタスク: python -m src.main --quick-test --llm anthropic
- 任意タスク: python -m src.main --task "..." --llm anthropic
- A/B 比較: --no-peer-review で Round 2 を skip
"""

import re
import traceback
from datetime import datetime
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .models import (
    MAX_SECTION_MARKDOWN_LENGTH_ROUND2,
    MultiPathPeerReviewState,
    SynthesizerOutput,
    _PathLLMOutput,
    _PathRound2LLMOutput,
)
from .output_saver import save_output
from .perspectives import (
    DOMAIN_CONTEXT,
    MAX_ITERATIONS_PER_PATH,
    MIN_TAVILY_SCORE,
    NUM_PATHS,
    PERSPECTIVE_POOL,
    PathResult,
    PathSource,
    Perspective,
    PerspectiveID,
    SelectedPerspectives,
    format_perspective_pool_md,
    get_pool_summary,
    label_for_id,
)
from .search_tool import search_tool


# Round 2 (peer-review) の path executor のループ上限。
# Round 1 の MAX_ITERATIONS_PER_PATH (= 2) とは独立した定数として定義する。
# 現状はたまたま同じ値だが、設計意図としては別物 (Round 1 = 新規調査、
# Round 2 = 矛盾解消) であり、独立して調整可能にしておく。
#
# この値が Tavily 呼び出し回数の構造的上限を兼ねる
# (制約の 4 段階モデル 段階 3: コードループ上限)。
MAX_ITERATIONS_ROUND2 = 2


# ===== LLM ファクトリ =====


def _create_heavy_process_llm(
    provider: str,
    settings,
) -> BaseChatModel:
    """Synthesizer 用の Heavy LLM (timeout=300s, max_tokens=8000)。"""
    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_smart_model,
            temperature=settings.temperature,
            timeout=300,
            max_retries=2,
            max_tokens=8000,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=settings.anthropic_smart_model,
            temperature=settings.temperature,
            timeout=300,
            max_retries=2,
            max_tokens=8000,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ===== サブエージェント: PerspectiveSelector =====


class PerspectiveSelector:
    """ユーザー目標から curated list の 4 視点を選定する (Light LLM)。"""

    PROMPT_TEMPLATE = (
        "あなたは **気象 × AI 領域** のリサーチ設計エージェントです。"
        "ユーザーの調査目標に対し、curated list から最も適切な"
        "**{n} つの視点** を選定してください。\n\n"
        "{domain_context}\n\n"
        "---\n\n"
        "ユーザー目標: {query}\n"
        "現在の日付: {current_date}\n\n"
        "## 視点候補 ({pool_size} 個)\n\n"
        "視点には Core (中核 3 視点: ai_tech, data_infra, genai_application) と"
        "補完 (5 視点) があります。クエリの性質に応じて選定してください。\n\n"
        "{perspectives_md}\n\n"
        "## 選定の目標\n\n"
        "選んだ {n} 視点が組み合わさることで、ユーザー目標を **多角的に深掘り** "
        "できる構成を作る。各視点が異なる側面を担当し、合計でユーザー目標の"
        "全体像をカバーする。\n\n"
        "## 評価軸\n\n"
        "1. **関連度**: 各視点が、ユーザー目標に対して有意な情報・洞察を"
        "提供できるか。description を読み、目標達成への貢献を判断する。\n"
        "2. **補完性**: 選んだ {n} 視点が互いに異なる側面を担当しているか。"
        "同じ問いに対して同じ答えを出す視点を 2 つ選ばない。\n"
        "3. **Core の優先**: 気象 × AI のクエリでは Core 3 視点 "
        "(ai_tech, data_infra, genai_application) のうち 2-3 個が選ばれることが多い。"
        "残り 1-2 枠を補完視点で埋めるイメージ。"
        "ただしクエリが純粋に制度・倫理・歴史・社会寄りなら、"
        "補完視点中心の選定もあり得る。query から判断する。\n\n"
        "## 厳格なルール\n\n"
        "1. **必ず {n} 個** 選定する。{n_minus_one} 個・{n_plus_one} 個は不可。\n"
        "2. **ID の重複禁止**。\n"
        "3. **無理に当てはめない**: 関連の薄い視点を「{n} 個埋めるため」に"
        "選んではならない。\n"
        "4. **rationale の必須構造**:\n"
        "   - 冒頭 1-2 文: なぜこの {n} 視点の組み合わせを選んだか(全体方針)\n"
        "   - 各視点について 1 文ずつ: その視点が何を担当し、"
        "Tier 1 機関 (ECMWF / NOAA / DeepMind) のどんな情報源に当たることになりそうか\n"
        "   - 合計 200-400 文字"
    )

    def __init__(
        self,
        llm: BaseChatModel,
        pool: list[Perspective] = PERSPECTIVE_POOL,
        num_paths: int = NUM_PATHS,
    ):
        self.llm = llm
        self.pool = pool
        self.num_paths = num_paths

    def run(self, query: str) -> SelectedPerspectives:
        current_date = datetime.now().strftime("%Y-%m-%d")
        perspectives_md = format_perspective_pool_md(self.pool)
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        chain = prompt | self.llm.with_structured_output(SelectedPerspectives)
        return chain.invoke(
            {
                "query": query,
                "current_date": current_date,
                "n": self.num_paths,
                "n_minus_one": self.num_paths - 1,
                "n_plus_one": self.num_paths + 1,
                "pool_size": len(self.pool),
                "perspectives_md": perspectives_md,
                "domain_context": DOMAIN_CONTEXT,
            }
        )


# ===== サブエージェント: PathExecutor (Round 1) =====


class PathExecutor:
    """1 視点について手動ツールループ + 構造化執筆を実行する (Light LLM、Round 1 用)。

    制約の 4 段階モデル 適用:
    - 段階 1 プロンプト: 文字数厳守、include_domains ガイダンス
    - 段階 2 Pydantic: max_length=1700、PerspectiveID Literal
    - 段階 4 コード: Tavily score < MIN_TAVILY_SCORE を自動除外
    """

    SYSTEM_PROMPT_TEMPLATE = (
        "あなたは気象 × AI 領域の **複数視点研究** におけるリサーチエージェントです。"
        "全体の調査目標について、{n_total} 視点のうち **1 つの視点** を担当します。\n\n"
        "## ユーザーの調査目標\n"
        "{query}\n\n"
        "## あなたが担当する視点\n"
        "ID: `{perspective_id}`\n"
        "Label: {perspective_label}\n"
        "Description:\n"
        "{perspective_description}\n\n"
        "{domain_context}\n\n"
        "---\n\n"
        "## 責務\n"
        "1. **担当視点の枠内** で、ユーザー目標を考察するセクションを執筆する\n"
        "2. **他視点の領域** (description の \"※...\" 部分) には踏み込まない\n"
        "3. **Tier 1 機関** (ECMWF / NOAA / Google DeepMind) の公式情報・論文を最優先\n"
        "4. **2022 年以前の手法は AI 文脈への接続点としてのみ扱う**\n\n"
        "## 検索ルール (厳守)\n"
        "1. tavily_search は **最大 {max_iterations} 回** まで\n"
        "2. 1 回目: 担当視点と目標の交点を広く検索\n"
        "3. 2 回目 (必要時のみ): 1 回目で得た固有名詞・公式リンクで深掘り\n"
        "4. **include_domains の指定**: 上記「Tavily 検索ガイドライン」を厳守。\n"
        "   - 推奨ドメイン (ecmwf.int, noaa.gov, deepmind.google, blog.google, "
        "registry.opendata.aws 等) を活用\n"
        "   - **arxiv.org を include_domains に含めない** "
        "(preprint 品質ばらつき大)\n"
        "   - genai_application 視点では anthropic.com, python.langchain.com, "
        "github.com 等も活用可\n"
        "5. **コード側のフィルタ**: 検索結果はコード側で score < {min_tavily_score} の結果が"
        "自動除外される。ただし、これは下限フィルタに過ぎない (下記「品質判断」を参照)。\n"
        "6. 検索結果が部分的でも、得られた範囲で執筆する。"
        "情報不足は section 内に「公開情報では確認できず」と明記する。\n\n"
        "## Tavily 結果の品質判断 (重要)\n"
        "Tavily の挙動について以下の事実に注意し、引用するソースを慎重に選別すること:\n"
        "- **`include_domains` は hint であり hard filter ではない**。"
        "指定外ドメインの結果も返ることがある (Tavily が「関連度が高い」と判断した場合)\n"
        "- **`score` は関連度であって信頼性ではない**。"
        "SEO 最適化されたサイトは高 score (0.99+) を取りやすい。"
        "Tier 1 機関の正規情報源は逆に score 0.5-0.7 程度に留まることが多い\n"
        "- **引用判断は score ではなく「発信元の権威性」で行う**:\n"
        "  - 最優先: Tier 1 機関 (ECMWF / NOAA / DeepMind / WMO / Microsoft Research / "
        "AWS Open Data Registry の公式ページ)\n"
        "  - 次点: 査読論文 (Nature / Science / 学会誌)、機関公式ブログ\n"
        "  - 補助: 信頼できる個人技術ブログ (内容が公式情報と整合する場合)\n"
        "  - **避ける**: 集約系・SEO 寄りのサイト "
        "(URL に `post/`, `blog/2026/` 等の汎用パターン、ドメインに company/ai/tech 等の汎用語を含む、"
        "出典が不明確な記事)\n"
        "- 同じ事実を複数の Tier 1 ソースが確認していれば、それを引用する。"
        "情報源を分散させて補強する\n"
        "- score が高いだけの理由で SEO サイトを引用しない。"
        "代わりに同じ情報を発信している Tier 1 ソースを探すか、その事実の引用を諦める\n\n"
        "## 文字数の厳守 (最重要制約)\n"
        "- section_markdown は **1,200-1,400 文字以内** に厳守する\n"
        "- **1,500 文字を超えてはならない** (上限 max_length=1700、超過は出力エラーとなる)\n"
        "- 多くの情報を盛り込むより、**視点の核心を凝縮** することを優先する\n"
        "- **サブセクションは 3 個以内** に絞る (各サブセクション 200-400 字程度)\n"
        "- 列挙が長くなる場合は要点のみに絞り、**詳細は出典 URL に委ねる**\n"
        "- (重要) 一覧表や全機関を網羅する記述は不要。代表例 2-3 個に絞る\n\n"
        "## 視点の参照表記 (重要)\n\n"
        "- 本文中で他視点を参照する際は、必ず perspective.label (日本語ラベル) を使用すること\n"
        "  例: 「AI/ML 技術観点」「データ・観測インフラ観点」「生成AI 活用観点」"
        "「国際比較・機関別比較観点」\n"
        "- perspective.id (内部識別子、例: `ai_tech`, `data_infra`, `genai_application`, "
        "`comparative_intl`, `meteorology`, `disaster_society`, `policy_ethics`, "
        "`history_evolution`) を本文に出してはならない。これは内部 ID であり、読者には意味不明\n"
        "- 「視点を参照」する代表的な表現:\n"
        "  - ✓ 「データ・観測インフラ観点が示すように…」\n"
        "  - ✓ 「(技術詳細は AI/ML 技術観点を参照)」\n"
        "  - ✗ 「data_infra 観点が指摘するように」 (id 表記、禁止)\n"
        "  - ✗ 「`ai_tech` 観点を参照」 (id 表記、禁止)\n"
        "- バッククォート `xxx` で id を囲む書き方も禁止\n\n"
        "## 現在時点と対象期間\n\n"
        "- 現在は **2026 年** であり、レポートを書く時点の最新は 2026 年の動向を含む\n"
        "- 対象期間は **2023 年から現在 (2026 年) まで** とし、すべての年を同等の重要度で扱うこと\n"
        "- 2025 年までの動向に偏ることなく、2026 年の最新の運用実績・新発表・利用状況等も"
        "積極的に取り込むこと\n"
        "- 2026 年の情報源は Tavily 検索結果に含まれている可能性が高い。検索クエリでも"
        "「2026」を含めることを検討する\n"
        "- 過去の出来事 (2023-2024 年) と現在進行中の動向 (2025-2026 年) の両方を"
        "バランスよく扱う\n\n"
        "## 日付・時期表現の規範\n\n"
        "固有名詞の日付・時期は **月単位 (年月)** までで記述するのが標準。"
        "日単位の精度は通常不要であり、視点間の不要な不整合を生む。\n\n"
        "### 月単位を使う対象 (デフォルト)\n\n"
        "- 機関のモデル運用化・正式リリース時期\n"
        "  例: 「ECMWF AIFS-ENS は 2025 年 7 月に運用化」\n"
        "- プロダクトの発表時期\n"
        "  例: 「Google WeatherNext 2 は 2025 年 11 月に発表」\n"
        "- データセットの公開時期\n"
        "  例: 「Anemoi training-ready ERA5 は 2025 年に公開」\n\n"
        "### 日単位を使う対象 (例外、限定的に許容)\n\n"
        "以下のケースのみ日単位を許容する:\n"
        "- 論文の発表日 (DOI で確定する一次情報)\n"
        "  例: 「Lam et al., Science 2023 (doi:10.1126/science.adi2336)」\n"
        "- 公式文書の発行日 (引用の特定に必要な場合)\n"
        "- 1 日の意味を持つ特定のイベント (会議の開催日など)\n\n"
        "それ以外で日単位を書くと、他視点と細部で不整合が生じやすい。"
        "迷ったら月単位を選ぶこと。\n\n"
        "## 最終出力形式\n"
        "_PathLLMOutput 構造体として返す:\n"
        "- section_markdown: 担当視点に基づくセクション本文 "
        "(Markdown、見出し ## から開始、**1,200-1,400 字以内厳守**)\n"
        "- sources: 引用した Tavily 結果のリスト (title / url / snippet)"
    )

    FINAL_WRITE_INSTRUCTION = (
        "これまでに得た情報をもとに、担当視点のセクション本文と出典リストを"
        "_PathLLMOutput 構造で出力してください。\n\n"
        "**重要な制約**:\n"
        "- section_markdown は **1,200-1,400 字以内に厳守** "
        "(1,500 字を超えてはならない、超過するとスキーマ検証エラーになります)\n"
        "- 長くなりそうな場合は、列挙を要点のみに絞り、詳細は出典 URL に委ねてください\n"
        "- サブセクションは 3 個以内、各 200-400 字を目安にしてください\n"
        "- sources には実際に section 内で引用した Tavily 結果のみを含めてください\n"
        "- 情報が不足している項目は section 内で「公開情報では確認できず」と明記してください"
    )

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = MAX_ITERATIONS_PER_PATH,
        n_total: int = NUM_PATHS,
    ):
        self.llm_with_tools = llm.bind_tools([search_tool])
        self.llm = llm
        self.tool = search_tool
        self.max_iterations = max_iterations
        self.n_total = n_total

    def run(self, query: str, perspective: Perspective) -> PathResult:
        sys_content = self.SYSTEM_PROMPT_TEMPLATE.format(
            query=query,
            perspective_id=perspective.id,
            perspective_label=perspective.label,
            perspective_description=perspective.description,
            domain_context=DOMAIN_CONTEXT,
            max_iterations=self.max_iterations,
            n_total=self.n_total,
            min_tavily_score=MIN_TAVILY_SCORE,
        )
        messages = [
            SystemMessage(content=sys_content),
            HumanMessage(
                content=(
                    f"視点 `{perspective.id}` ({perspective.label}) を担当します。"
                    "必要な情報を tavily_search で集めてから、セクションを執筆してください。"
                )
            ),
        ]

        # Manual tool loop
        for iteration in range(self.max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            print(
                f"      [path:{perspective.id}] "
                f"iter {iteration + 1}/{self.max_iterations} "
                f"tool calls: {len(response.tool_calls)}"
            )

            for tool_call in response.tool_calls:
                if tool_call["name"] == self.tool.name:
                    tool_result = self.tool.invoke(tool_call["args"])

                    # Tavily score filter (制約の 4 段階モデル 段階 4)
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        original_count = len(tool_result["results"])
                        tool_result["results"] = [
                            r
                            for r in tool_result["results"]
                            if r.get("score", 0) >= MIN_TAVILY_SCORE
                        ]
                        dropped = original_count - len(tool_result["results"])
                        if dropped > 0:
                            print(
                                f"        [path:{perspective.id}] "
                                f"[filter] dropped {dropped}/{original_count} "
                                f"low-score results (< {MIN_TAVILY_SCORE})"
                            )

                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        )
                    )

        # Final structured write
        messages.append(HumanMessage(content=self.FINAL_WRITE_INSTRUCTION))
        output: _PathLLMOutput = self.llm.with_structured_output(_PathLLMOutput).invoke(
            messages
        )

        # Inject perspective info from authoritative source
        return PathResult(
            perspective_id=perspective.id,
            perspective_label=perspective.label,
            section_markdown=output.section_markdown,
            sources=output.sources,
            round=1,
        )


# ===== サブエージェント: PathExecutorRound2 =====


class PathExecutorRound2:
    """Round 2 の path executor (Light LLM)。

    Round 1 の独立執筆後、他視点の drafts を踏まえて自分の section を改訂する。

    手動ツールループパターン:
    - max_iterations までループ
    - 各 iter で llm_with_tools.invoke → tool_calls あれば実行
    - tool_calls なしで break
    - loop 後に FINAL_WRITE_INSTRUCTION + with_structured_output で最終出力

    制約の 4 段階モデル 適用:
    - 段階 1 プロンプト: 「Tavily は矛盾解消用途のみ、最大 2 回」を明示
    - 段階 2 Pydantic: max_length=1700, max_length=400 (revision_notes)
    - 段階 3 コード: MAX_ITERATIONS_ROUND2 (=2) でループ上限 (Round 1 とは独立した定数。Tavily 上限 2 を兼ねる)
    - 段階 4 コード: Tavily score < MIN_TAVILY_SCORE を自動除外
    """

    SYSTEM_PROMPT_TEMPLATE = (
        "あなたは気象 × AI 領域の **複数視点研究** におけるリサーチエージェントです。"
        "Round 1 では各視点が独立にレポートを執筆しました。"
        "あなたの Round 1 draft と他視点の drafts が以下に提示されます。\n\n"
        "Round 2 のあなたの責務: 他視点の drafts を踏まえて **自分の section だけ** を改訂する。\n\n"
        "## ユーザーの調査目標\n"
        "{query}\n\n"
        "## あなたが担当する視点\n"
        "ID: `{perspective_id}`\n"
        "Label: {perspective_label}\n"
        "Description:\n"
        "{perspective_description}\n\n"
        "{domain_context}\n\n"
        "---\n\n"
        "## あなたの Round 1 draft\n\n"
        "{own_draft_markdown}\n\n"
        "---\n\n"
        "## 他視点の drafts (perspective_id 順)\n\n"
        "{sibling_drafts_md}\n\n"
        "---\n\n"
        "## 改訂目標\n"
        "1. **冗長削減**: 他視点と同じ事実を同じ深さで扱っているなら、"
        "その視点を Core とする側に譲り、自分は別角度から触れるか省略する\n"
        "2. **空白補填**: 自分の視点で扱うべきだったが Round 1 で漏れた点を追加する\n"
        "3. **矛盾解消**: 数値・年代・固有名詞の食い違いがあれば再検証して整合させる\n"
        "4. **明示的 cross-reference**: 接続点を本文で示す "
        "(例:「データ観点が指摘するように」)\n"
        "5. **独自性保持**: 視点固有の angle は薄めない。他視点と均質化することは禁止\n\n"
        "## 制約\n"
        "- **改訂が不要なら revision_notes に exactly 'significant change not required' と書き、"
        "Round 1 draft を維持してよい**。強制的に改訂する必要はない。\n"
        "- tavily_search は **矛盾解消用途のみ、最大 {max_iterations} 回** まで。"
        "新規調査での使用は禁止。\n"
        "- 他視点の section は書き換えない (自分の section のみ)\n"
        "- referenced_siblings には実際に本文中で参照した他視点の id のみを列挙する\n\n"
        "## Tavily 結果の品質判断\n"
        "DOMAIN_CONTEXT 内の品質判断指針に従うこと:\n"
        "- include_domains は hint であり hard filter ではない\n"
        "- score は関連度であって信頼性ではない\n"
        "- 引用判断は score ではなく「発信元の権威性」で行う "
        "(Tier 1 機関 > 査読論文 > 公式ブログ > ...)\n"
        "- 集約系・SEO 寄りのサイトを引用しない\n\n"
        "## Round 2 の本質: 最小限の差分\n\n"
        "Round 2 はゼロから書き直すのではなく、**Round 1 draft への最小限の差分** を生成する。\n"
        "- Round 1 を維持するのが基本姿勢。改訂は「修正が必要な箇所のみ」をピンポイントに行う\n"
        "- 大幅な再構成・全面書き直しは禁止 (それは Round 1 の責務であり Round 2 の責務ではない)\n"
        "- 結果として Round 2 出力は Round 1 と概ね同じ長さに収まることが期待される\n"
        "- 文字数が大きく増える場合、それは「改訂」ではなく「書き直し」になっている兆候\n\n"
        "## 文字数の規範 (規律として遵守)\n\n"
        "絶対値 target ではなく **Round 1 を起点とした相対基準** で考えること:\n"
        "- section_markdown は **Round 1 の文字数 ± 200 字以内** が目安\n"
        "  例: Round 1 が 1,200 字なら Round 2 は 1,000-1,400 字\n"
        "  例: Round 1 が 1,500 字なら Round 2 は 1,300-1,700 字\n"
        "- 例外: Round 1 が短く (1,000 字未満) かつ重要な空白補填がある場合のみ "
        "1,500 字まで許容\n"
        "- いかなる場合も上限 max_length={max_section_length} 字を超えてはならない "
        "(構造的安全策、超過は出力エラー)\n"
        "- 「上限まで書いてよい」と解釈してはならない。上限は異常時の防御であって "
        "目標値ではない\n\n"
        "## 視点の参照表記 (重要)\n\n"
        "- 本文中で他視点を参照する際は、必ず perspective.label (日本語ラベル) を使用すること\n"
        "  例: 「AI/ML 技術観点」「データ・観測インフラ観点」「生成AI 活用観点」"
        "「国際比較・機関別比較観点」\n"
        "- perspective.id (内部識別子、例: `ai_tech`, `data_infra`, `genai_application`, "
        "`comparative_intl`, `meteorology`, `disaster_society`, `policy_ethics`, "
        "`history_evolution`) を本文に出してはならない。これは内部 ID であり、読者には意味不明\n"
        "- 「視点を参照」する代表的な表現:\n"
        "  - ✓ 「データ・観測インフラ観点が示すように…」\n"
        "  - ✓ 「(技術詳細は AI/ML 技術観点を参照)」\n"
        "  - ✗ 「data_infra 観点が指摘するように」 (id 表記、禁止)\n"
        "  - ✗ 「`ai_tech` 観点を参照」 (id 表記、禁止)\n"
        "- バッククォート `xxx` で id を囲む書き方も禁止\n\n"
        "## 現在時点と対象期間\n\n"
        "- 現在は **2026 年** であり、レポートを書く時点の最新は 2026 年の動向を含む\n"
        "- 対象期間は **2023 年から現在 (2026 年) まで** とし、すべての年を同等の重要度で扱うこと\n"
        "- 2025 年までの動向に偏ることなく、2026 年の最新の運用実績・新発表・利用状況等も"
        "積極的に取り込むこと\n"
        "- 2026 年の情報源は Tavily 検索結果に含まれている可能性が高い。検索クエリでも"
        "「2026」を含めることを検討する\n"
        "- 過去の出来事 (2023-2024 年) と現在進行中の動向 (2025-2026 年) の両方を"
        "バランスよく扱う\n\n"
        "## 日付・時期表現の規範\n\n"
        "固有名詞の日付・時期は **月単位 (年月)** までで記述するのが標準。"
        "日単位の精度は通常不要であり、視点間の不要な不整合を生む。\n\n"
        "### 月単位を使う対象 (デフォルト)\n\n"
        "- 機関のモデル運用化・正式リリース時期\n"
        "  例: 「ECMWF AIFS-ENS は 2025 年 7 月に運用化」\n"
        "- プロダクトの発表時期\n"
        "  例: 「Google WeatherNext 2 は 2025 年 11 月に発表」\n"
        "- データセットの公開時期\n"
        "  例: 「Anemoi training-ready ERA5 は 2025 年に公開」\n\n"
        "### 日単位を使う対象 (例外、限定的に許容)\n\n"
        "以下のケースのみ日単位を許容する:\n"
        "- 論文の発表日 (DOI で確定する一次情報)\n"
        "  例: 「Lam et al., Science 2023 (doi:10.1126/science.adi2336)」\n"
        "- 公式文書の発行日 (引用の特定に必要な場合)\n"
        "- 1 日の意味を持つ特定のイベント (会議の開催日など)\n\n"
        "それ以外で日単位を書くと、他視点と細部で不整合が生じやすい。"
        "迷ったら月単位を選ぶこと。\n\n"
        "### Round 2 における追加指針\n\n"
        "- 他視点の draft に書かれた日付に言及する場合、その日付を **月単位に丸めて** 記述すること\n"
        "  例: 他視点が「12 月 17 日」と書いていても、自分の section では「12 月」とする\n"
        "- 他視点の draft で日単位の記述があっても、自分の section ではより粗い粒度を選ぶことで、"
        "視点間の不整合を未然に防ぐ\n"
        "- 日付に関する矛盾解消で Tavily を使う場合も、解消後の本文記述は月単位とする\n\n"
        "## 最終出力形式\n"
        "_PathRound2LLMOutput 構造体として返す:\n"
        "- section_markdown: 改訂後の section 本文 (改訂不要なら Round 1 と同じ)\n"
        "- sources: 改訂後の引用ソース一覧\n"
        "- revision_notes: 何をどう変えたか / なぜ (1-3 文、改訂不要なら exactly "
        "'significant change not required')\n"
        "- referenced_siblings: 本文中で参照した他視点の id のリスト"
    )

    FINAL_WRITE_INSTRUCTION = (
        "これまでに得た情報をもとに、改訂後のセクション本文と revision_notes を"
        "_PathRound2LLMOutput 構造で出力してください。\n\n"
        "**Round 2 の本質を再確認**:\n"
        "- Round 1 への最小限の差分を生成する\n"
        "- ゼロから書き直すのではなく、修正が必要な箇所のみピンポイントに変更\n"
        "- 改訂が不要と判断した場合は revision_notes に "
        "'significant change not required' と書き、Round 1 と同じ内容で出力\n\n"
        "**文字数の規範**:\n"
        "- section_markdown は Round 1 の文字数 ± 200 字以内が目安\n"
        f"- 上限は {MAX_SECTION_MARKDOWN_LENGTH_ROUND2} 字 "
        "(これを超えるとスキーマ検証エラー、ただし上限は目標値ではなく異常時の防御)\n\n"
        "referenced_siblings には本文中で実際に明示的に参照した他視点の id のみを含めてください "
        "(参照しなかった場合は空リスト)"
    )

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = MAX_ITERATIONS_ROUND2,  # = 2、Round 1 とは独立した定数
        n_total: int = NUM_PATHS,
    ):
        self.llm_with_tools = llm.bind_tools([search_tool])
        self.llm = llm
        self.tool = search_tool
        self.max_iterations = max_iterations
        self.n_total = n_total

    def run(
        self,
        query: str,
        perspective: Perspective,
        own_draft: PathResult,
        sibling_drafts: list[PathResult],
    ) -> PathResult:
        # sibling drafts を perspective_id 順に整形
        sorted_siblings = sorted(sibling_drafts, key=lambda r: r.perspective_id)
        sibling_drafts_md = "\n\n".join(
            f"### `{s.perspective_id}` — {s.perspective_label}\n\n{s.section_markdown}"
            for s in sorted_siblings
        )

        sys_content = self.SYSTEM_PROMPT_TEMPLATE.format(
            query=query,
            perspective_id=perspective.id,
            perspective_label=perspective.label,
            perspective_description=perspective.description,
            domain_context=DOMAIN_CONTEXT,
            max_iterations=self.max_iterations,
            max_section_length=MAX_SECTION_MARKDOWN_LENGTH_ROUND2,
            own_draft_markdown=own_draft.section_markdown,
            sibling_drafts_md=sibling_drafts_md,
        )
        messages = [
            SystemMessage(content=sys_content),
            HumanMessage(
                content=(
                    f"視点 `{perspective.id}` ({perspective.label}) の改訂を開始します。"
                    "他視点の drafts を踏まえて自分の section を見直してください。"
                    "矛盾解消が必要な場合のみ tavily_search を使ってください "
                    f"(最大 {self.max_iterations} 回)。"
                )
            ),
        ]

        # Manual tool loop
        for iteration in range(self.max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            print(
                f"      [path_round2:{perspective.id}] "
                f"iter {iteration + 1}/{self.max_iterations} "
                f"tool calls: {len(response.tool_calls)}"
            )

            for tool_call in response.tool_calls:
                if tool_call["name"] == self.tool.name:
                    tool_result = self.tool.invoke(tool_call["args"])

                    # Tavily score filter (制約の 4 段階モデル 段階 4)
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        original_count = len(tool_result["results"])
                        tool_result["results"] = [
                            r
                            for r in tool_result["results"]
                            if r.get("score", 0) >= MIN_TAVILY_SCORE
                        ]
                        dropped = original_count - len(tool_result["results"])
                        if dropped > 0:
                            print(
                                f"        [path_round2:{perspective.id}] "
                                f"[filter] dropped {dropped}/{original_count} "
                                f"low-score results (< {MIN_TAVILY_SCORE})"
                            )

                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        )
                    )

        # Final structured write (loop の外)
        messages.append(HumanMessage(content=self.FINAL_WRITE_INSTRUCTION))
        try:
            output: _PathRound2LLMOutput = self.llm.with_structured_output(
                _PathRound2LLMOutput
            ).invoke(messages)
        except Exception as e:
            # フォールバック: own_draft をそのまま返す
            print(
                f"      [path_round2:{perspective.id}] "
                f"FALLBACK: structured output failed ({type(e).__name__}: {e}), "
                f"keeping Round 1 draft"
            )
            return own_draft.model_copy(update={
                "round": 2,
                "revision_notes": f"fallback: structured output failed ({type(e).__name__})",
            })

        # Inject perspective info from authoritative source
        return PathResult(
            perspective_id=perspective.id,
            perspective_label=perspective.label,
            section_markdown=output.section_markdown,
            sources=output.sources,
            round=2,
            referenced_siblings=output.referenced_siblings,
            revision_notes=output.revision_notes,
        )


# ===== サブエージェント: CrossPathSynthesizer =====


class CrossPathSynthesizer:
    """4 つの PathResult を統合する Heavy LLM (中庸統合 β 型)。

    入力切替 (Round 1 vs Round 2) は呼び出し側 (_synthesizer_node) で行う。
    """

    PROMPT_TEMPLATE = (
        "あなたは気象 × AI 領域の **複数視点研究** におけるシンセサイザー (統合者) です。"
        "ユーザーの調査目標について、{n_paths} つの視点からの調査結果が既に得られています。"
        "あなたの責務は、これらを統合する **イントロ・視点間観察・結論** を生成することです。\n\n"
        "{domain_context}\n\n"
        "---\n\n"
        "## ユーザーの調査目標\n"
        "{query}\n\n"
        "## 担当した {n_paths} つの視点\n\n"
        "{perspectives_md}\n\n"
        "## 各視点のセクション本文 (これは既に確定しています、rewrite 不要)\n\n"
        "{path_sections_md}\n\n"
        "---\n\n"
        "## あなたの生成タスク\n\n"
        "**重要**: 上記の各セクション本文を **rewrite しないでください**。"
        "あなたが生成するのは以下の 3 フィールドのみです:\n\n"
        "1. **introduction (200-400 字)**: ユーザー目標と {n_paths} 視点の構成を"
        "説明する導入。\n"
        "   - なぜこの {n_paths} 視点で目標を多角的にカバーできるのか\n"
        "   - 読者がレポートを読み始めるためのガイダンス\n\n"
        "2. **cross_path_observations (300-500 字)**: 視点間の横断的観察。"
        "**multi_path 設計の核心**。以下を扱う:\n"
        "   - **一致点**: 複数視点が同じ結論を支持している部分\n"
        "     例: 'AI/ML 技術観点とデータ・観測インフラ観点の両方が ERA5 の中心性を強調...'\n"
        "   - **相補点**: 視点間で補完的に組み合わさる情報\n"
        "     例: '気象学・地球科学観点の物理基盤理解が AI/ML 技術観点の手法選択を裏付ける...'\n"
        "   - **緊張点**: 見方が分かれる、または議論の余地がある部分\n"
        "     例: '国際比較・機関別比較観点では欧米先行を強調するが、政策・倫理・ガバナンス観点では...'\n"
        "   どの視点が何を述べているかは、下記「## 視点の参照表記 (重要)」に従い"
        "**必ず日本語ラベルで明示** しながら記述すること。\n\n"
        "3. **conclusion (200-300 字)**: 全体結論と示唆。\n"
        "   - {n_paths} 視点を経て見えてきた重要ポイントを凝縮\n"
        "   - ユーザー目標への直接的な答え\n\n"
        "## 視点の参照表記 (重要)\n\n"
        "- 本文中で各視点を参照する際は、必ず perspective.label (日本語ラベル、例: "
        "「AI/ML 技術観点」「データ・観測インフラ観点」「生成AI 活用観点」"
        "「国際比較・機関別比較観点」) を使用すること\n"
        "- perspective.id (内部識別子、例: `ai_tech`, `data_infra`, `genai_application`, "
        "`comparative_intl`) を本文に出してはならない。これは内部 ID であり、読者には意味不明\n"
        "- 一致点・相補点・緊張点を述べる際も上記原則を遵守する\n\n"
        "## 厳守事項\n"
        "- 各フィールドは **指定文字数を厳守** (max_length 制約あり、超過は検証エラー)\n"
        "- 各セクション本文の rewrite や要約は不要 (それらは既に確定している)\n"
        "- 引用 URL の追加は不要 (出典は別途コードで集約される)\n"
        "- cross_path_observations では各視点を **日本語ラベル** で明示する "
        "(perspective.id を本文に出さない、上記「## 視点の参照表記」参照)"
    )

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(
        self,
        query: str,
        path_results: list[PathResult],
    ) -> SynthesizerOutput:
        path_sections_md = "\n\n".join(
            f"### `{r.perspective_id}` — {r.perspective_label}\n\n{r.section_markdown}"
            for r in path_results
        )
        perspectives_md = "\n".join(
            f"- `{r.perspective_id}` — {r.perspective_label}" for r in path_results
        )

        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        chain = prompt | self.llm.with_structured_output(SynthesizerOutput)
        return chain.invoke(
            {
                "query": query,
                "n_paths": len(path_results),
                "perspectives_md": perspectives_md,
                "path_sections_md": path_sections_md,
                "domain_context": DOMAIN_CONTEXT,
            }
        )


# ===== Graph nodes =====


def _selector_node(
    state: MultiPathPeerReviewState, selector: PerspectiveSelector
) -> dict:
    """selector を実行し、4 視点と rationale を State に返す。"""
    print(f"  [selector] running for query: {state.query[:60]}...")
    result = selector.run(state.query)
    perspectives = []
    for pid in result.selected_ids:
        p = next((x for x in PERSPECTIVE_POOL if x.id == pid), None)
        if p is not None:
            perspectives.append(p)
    print(f"  [selector] selected: {[p.id for p in perspectives]}")
    return {
        "perspectives": perspectives,
        "selector_rationale": result.rationale,
    }


def _fan_out_to_paths(state: MultiPathPeerReviewState) -> list[Send]:
    """Send API で N 並列 Round 1 パスへ動的 fan-out。"""
    print(f"  [fan-out:R1] dispatching {len(state.perspectives)} parallel paths")
    return [
        Send(
            "path_executor",
            {
                "perspective_id": p.id,
                "perspective_label": p.label,
                "perspective_description": p.description,
                "query": state.query,
            },
        )
        for p in state.perspectives
    ]


def _path_executor_node(input_state: dict, path_executor: PathExecutor) -> dict:
    """1 視点について PathExecutor (Round 1) を実行し、PathResult を返す。"""
    perspective = Perspective(
        id=input_state["perspective_id"],
        label=input_state["perspective_label"],
        description=input_state["perspective_description"],
    )
    print(f"    [path:{perspective.id}] start")
    result = path_executor.run(query=input_state["query"], perspective=perspective)
    print(
        f"    [path:{perspective.id}] done: "
        f"{len(result.section_markdown)} chars, {len(result.sources)} sources"
    )
    return {"path_results": [result]}


# ===== グラフノード =====


def _round2_router_node(state: MultiPathPeerReviewState) -> dict:
    """No-op gate. Round 1 fan-in 後の conditional_edges を hang するための空ノード。

    LangGraph の conditional_edges は単一ノードに hang する仕様のため、Round 1 の
    N 並列 fan-in 後に判定を入れるには空の中継ノードが必要。
    """
    return {}  # state は変更しない


def _decide_round2(
    state: MultiPathPeerReviewState,
) -> list[Send] | str:
    """Round 2 を回すか synthesizer に直行するかを判定。

    返り値:
    - list[Send]: Round 2 を回す (path_executor_round2 へ N 並列 fan-out)
    - "synthesizer": Round 2 を skip して synthesizer へ直行
    """
    # Toggle off
    if not state.peer_review_enabled:
        print(f"  [round2_router] peer_review disabled, skip Round 2")
        return "synthesizer"

    # 部分失敗時のフォールバック
    if len(state.path_results) != len(state.perspectives):
        print(
            f"  [round2_router] partial Round 1 failure: "
            f"{len(state.path_results)}/{len(state.perspectives)} paths, skip Round 2"
        )
        return "synthesizer"

    print(
        f"  [round2_router] dispatching {len(state.path_results)} parallel Round 2 paths"
    )

    # perspective lookup map
    perspective_by_id = {p.id: p for p in state.perspectives}

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
                    "perspective_description": perspective_by_id[
                        own.perspective_id
                    ].description,
                    "query": state.query,
                    "own_draft": own.model_dump(),
                    "sibling_drafts": [s.model_dump() for s in siblings],
                },
            )
        )
    return sends


def _path_executor_round2_node(
    input_state: dict, executor_round2: PathExecutorRound2
) -> dict:
    """1 視点について PathExecutorRound2 を実行し、改訂版 PathResult を返す。"""
    perspective = Perspective(
        id=input_state["perspective_id"],
        label=input_state["perspective_label"],
        description=input_state["perspective_description"],
    )
    own_draft = PathResult(**input_state["own_draft"])
    sibling_drafts = [PathResult(**s) for s in input_state["sibling_drafts"]]

    print(f"    [path_round2:{perspective.id}] start")
    result = executor_round2.run(
        query=input_state["query"],
        perspective=perspective,
        own_draft=own_draft,
        sibling_drafts=sibling_drafts,
    )
    notes_preview = (result.revision_notes[:60] + "...") if len(
        result.revision_notes
    ) > 60 else result.revision_notes
    print(
        f"    [path_round2:{perspective.id}] done: "
        f"{len(result.section_markdown)} chars, "
        f"refs={result.referenced_siblings}, "
        f"notes='{notes_preview}'"
    )
    return {"path_results_revised": [result]}


def _select_synthesizer_inputs(
    state: MultiPathPeerReviewState,
) -> list[PathResult]:
    """synthesizer / assembly が読む path_results を選ぶ。

    - peer_review_enabled かつ Round 2 が完走 (件数一致) → revised
    - それ以外 (toggle off, 部分失敗等) → Round 1 へフォールバック
    """
    if (
        state.peer_review_enabled
        and len(state.path_results_revised) == len(state.perspectives)
    ):
        return state.path_results_revised
    return state.path_results


def _synthesizer_node(
    state: MultiPathPeerReviewState, synthesizer: CrossPathSynthesizer
) -> dict:
    """Heavy LLM で intro / cross / conclusion を生成する。

    peer_review が有効かつ Round 2 が完走したら revised を、そうでなければ
    Round 1 を使う (フォールバック)。
    """
    inputs = _select_synthesizer_inputs(state)
    print(
        f"  [synthesizer] running on {len(inputs)} path results "
        f"(round={inputs[0].round if inputs else '?'})"
    )
    output = synthesizer.run(state.query, inputs)
    print(
        f"  [synthesizer] done: intro {len(output.introduction)} / "
        f"cross {len(output.cross_path_observations)} / "
        f"conclusion {len(output.conclusion)} chars"
    )
    return {"synthesizer_output": output}


def _assembly_node(state: MultiPathPeerReviewState) -> dict:
    """Synthesizer 出力と PathResults をコード側で組み立てて final_report を生成する。"""
    print(f"  [assembly] building final report")

    syn = state.synthesizer_output
    if syn is None:
        return {"final_report": "(synthesizer output missing)"}

    # synthesizer と同じ入力切替 (Round 2 完走時は revised、それ以外は Round 1)
    inputs = _select_synthesizer_inputs(state)

    # 各セクション本文 (順序を perspectives と揃える)
    perspective_order = [p.id for p in state.perspectives]
    ordered_results = sorted(
        inputs,
        key=lambda r: (
            perspective_order.index(r.perspective_id)
            if r.perspective_id in perspective_order
            else 999
        ),
    )
    sections_text = "\n\n".join(r.section_markdown for r in ordered_results)

    # 出典統合: URL で dedup、引用元視点を併記
    seen_urls: set[str] = set()
    unique_sources: list[tuple[PathSource, str]] = []
    for r in ordered_results:
        for s in r.sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                unique_sources.append((s, r.perspective_id))
    sources_md = "\n".join(
        f"- [{s.title}]({s.url}) (引用: `{pid}`)"
        for s, pid in unique_sources
    )
    if not sources_md:
        sources_md = "(該当なし)"

    final = (
        f"# {state.query}\n\n"
        f"## はじめに\n\n{syn.introduction}\n\n"
        f"---\n\n"
        f"{sections_text}\n\n"
        f"---\n\n"
        f"## 視点間の横断的観察\n\n{syn.cross_path_observations}\n\n"
        f"## 結論\n\n{syn.conclusion}\n\n"
        f"---\n\n"
        f"## 出典\n\n{sources_md}\n"
    )
    print(f"  [assembly] final_report: {len(final)} chars")
    return {"final_report": final}


# ===== Orchestrator =====


class MultiPathPeerReview:
    """multi_path_peer_review のメインクラス。"""

    def __init__(self, light_llm: BaseChatModel, heavy_llm: BaseChatModel):
        self.selector = PerspectiveSelector(llm=light_llm)
        self.path_executor = PathExecutor(llm=light_llm)
        self.path_executor_round2 = PathExecutorRound2(llm=light_llm)
        self.synthesizer = CrossPathSynthesizer(llm=heavy_llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        """グラフを構築する。

        START → selector → (Send fan-out) → path_executor × N
        → round2_router → (conditional fan-out or skip)
          ├ path_executor_round2 × N → synthesizer
          └ synthesizer (skip)
        → assembly → END
        """
        graph = StateGraph(MultiPathPeerReviewState)
        graph.add_node("selector", lambda s: _selector_node(s, self.selector))
        graph.add_node(
            "path_executor", lambda s: _path_executor_node(s, self.path_executor)
        )
        graph.add_node("round2_router", _round2_router_node)
        graph.add_node(
            "path_executor_round2",
            lambda s: _path_executor_round2_node(s, self.path_executor_round2),
        )
        graph.add_node(
            "synthesizer", lambda s: _synthesizer_node(s, self.synthesizer)
        )
        graph.add_node("assembly", _assembly_node)

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
        graph.add_edge("synthesizer", "assembly")
        graph.add_edge("assembly", END)

        return graph.compile()

    def run(self, query: str, peer_review_enabled: bool = True) -> dict:
        """グラフを実行し、final state (dict) を返す。"""
        initial = MultiPathPeerReviewState(
            query=query, peer_review_enabled=peer_review_enabled
        )
        return self.graph.invoke(initial)


# ===== CLI エントリポイント =====


DEMO_TASK = (
    "GraphCast 登場以降の AI 気象予報モデルの技術系譜と、"
    "ECMWF AIFS / NOAA GraphCastGFS / Google WeatherNext の最新動向について整理してください。"
)


def main():
    import argparse

    from .settings import Settings, get_llm

    parser = argparse.ArgumentParser(
        description=(
            "multi_path_peer_review — 多経路 peer-review 気象 × AI リサーチエージェント"
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="調査目標 (省略時は --quick-test 必須)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="anthropic",
        choices=["openai", "anthropic"],
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="デモタスクで起動 (--task 省略可能)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "中間出力 (実行情報 / Selector の選定 / Synthesizer 中間出力 / "
            "Round 1 視点別素材 / Round 2 視点別素材 / Round 1-2 diff サマリ) "
            "も保存する (デバッグ用)。デフォルトは最終レポートのみ保存。"
        ),
    )
    parser.add_argument(
        "--no-peer-review",
        action="store_true",
        help=(
            "peer-review round (Round 2) を実行せず Round 1 のみで完結。"
            "A/B 比較用。"
        ),
    )
    args = parser.parse_args()

    if args.task is None and not args.quick_test:
        parser.error("--task または --quick-test のどちらかが必要です")
    task = args.task if args.task is not None else DEMO_TASK
    peer_review_enabled = not args.no_peer_review

    settings = Settings()
    light_llm = get_llm(provider=args.llm, settings=settings)
    heavy_llm = _create_heavy_process_llm(args.llm, settings)
    model_name = (
        settings.anthropic_smart_model
        if args.llm == "anthropic"
        else settings.openai_smart_model
    )

    print("=== Run Configuration (multi_path_peer_review) ===")
    print(f"  Task:           {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  LLM:            {args.llm} / {model_name}")
    print(f"  Pool:           {get_pool_summary()}")
    print(f"  N paths:        {NUM_PATHS} (parallel via Send API)")
    if peer_review_enabled:
        print(f"  Peer Review:    ENABLED (Round 2 will run)")
        print(
            f"  Round 2:        max {MAX_ITERATIONS_ROUND2} iterations per path "
            f"(= max {MAX_ITERATIONS_ROUND2} Tavily calls)"
        )
    else:
        print(f"  Peer Review:    DISABLED (Round 1 のみ)")
    print(f"  Section target: ~1,200-1,400 chars/path")
    print(f"  Tavily filter:  score >= {MIN_TAVILY_SCORE}")
    print(f"  Light LLM:      timeout=120s")
    print(f"  Heavy LLM:      timeout=300s, max_tokens=8000")
    if peer_review_enabled:
        print(f"  Expected cost:  ~$0.50-$0.80 (with peer_review)")
        print(f"  Expected time:  ~8-12 minutes")
    else:
        print(f"  Expected cost:  ~$0.25-$0.40 (no peer_review)")
        print(f"  Expected time:  ~5-8 minutes")
    print(
        f"  Output mode:    "
        f"{'verbose (最終レポート + 中間出力)' if args.verbose else 'minimal (最終レポートのみ)'}"
    )
    print(f"  Stop anytime:   Ctrl+C で中断可能")
    print()

    print("[INFO] Starting multi_path_peer_review graph...")
    print()
    start_time = datetime.now()

    try:
        orchestrator = MultiPathPeerReview(
            light_llm=light_llm, heavy_llm=heavy_llm
        )
        final_state = orchestrator.run(task, peer_review_enabled=peer_review_enabled)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return

    elapsed = (datetime.now() - start_time).total_seconds()

    # Console summary
    perspectives = final_state["perspectives"]
    path_results: list[PathResult] = final_state["path_results"]
    path_results_revised: list[PathResult] = final_state.get(
        "path_results_revised", []
    )
    synth = final_state["synthesizer_output"]
    final_report = final_state["final_report"]

    # perspective_id -> Round 1 PathResult の lookup
    r1_by_id = {r.perspective_id: r for r in path_results}
    r2_by_id = {r.perspective_id: r for r in path_results_revised}

    print()
    print("=== Summary ===")
    print(f"  Total elapsed:    {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Selected paths:   {[p.id for p in perspectives]}")
    print(f"  Path results R1:  {len(path_results)}")
    if peer_review_enabled:
        print(
            f"  Path results R2:  {len(path_results_revised)} "
            f"(peer_review enabled)"
        )
        if path_results_revised:
            print(f"  Round 2 changes:")
            # perspective 順で表示
            for p in perspectives:
                r1 = r1_by_id.get(p.id)
                r2 = r2_by_id.get(p.id)
                if r2 is None:
                    print(f"    - {p.id}: (no Round 2 result)")
                    continue
                r1_len = len(r1.section_markdown) if r1 else 0
                r2_len = len(r2.section_markdown)
                changed_marker = (
                    " (unchanged)"
                    if r2.revision_notes.strip()
                    == "significant change not required"
                    else ""
                )
                notes_preview = (
                    r2.revision_notes[:80] + "..."
                    if len(r2.revision_notes) > 80
                    else r2.revision_notes
                )
                print(
                    f"    - {p.id}: {r1_len} → {r2_len} chars{changed_marker}, "
                    f"refs={r2.referenced_siblings}, notes='{notes_preview}'"
                )
    else:
        print(f"  Path results R2:  -- (peer_review disabled)")

    if path_results:
        avg_len_r1 = sum(len(r.section_markdown) for r in path_results) / len(
            path_results
        )
        total_sources_r1 = sum(len(r.sources) for r in path_results)
        print(f"  Avg R1 section:   {avg_len_r1:.0f} chars")
        print(f"  Total R1 sources: {total_sources_r1}")
    if synth is not None:
        print(
            f"  Synthesizer out:  intro {len(synth.introduction)} / "
            f"cross {len(synth.cross_path_observations)} / "
            f"conclusion {len(synth.conclusion)} chars"
        )
    print(f"  Final report:     {len(final_report)} chars")

    # Save output
    sections: dict[str, str] = {}

    # [1] 最終レポート (常に最初・常に保存): 本成果物
    sections["最終レポート"] = final_report

    # [2-N] デバッグ用中間出力 (--verbose のときだけ保存)
    if args.verbose:
        sections["(デバッグ) 実行情報"] = (
            f"- LLM: {args.llm} / {model_name}\n"
            f"- N paths: {NUM_PATHS}\n"
            f"- Pool: {get_pool_summary()}\n"
            f"- Peer review: {'ENABLED' if peer_review_enabled else 'DISABLED'}\n"
            f"- Round 2 max iterations: {MAX_ITERATIONS_ROUND2}\n"
            f"- Tavily filter: score >= {MIN_TAVILY_SCORE}\n"
            f"- Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)\n"
            f"- Run timestamp: {datetime.now().isoformat()}"
        )

        sections["(デバッグ) Selector の選定"] = (
            "### 選定された視点\n\n"
            + "\n".join(f"- `{p.id}` — {p.label}" for p in perspectives)
            + "\n\n### Rationale\n\n"
            + final_state.get("selector_rationale", "")
        )

        if synth is not None:
            sections["(デバッグ) Synthesizer 中間出力"] = (
                f"### Introduction ({len(synth.introduction)} chars)\n\n"
                f"{synth.introduction}\n\n"
                f"### Cross-Path Observations "
                f"({len(synth.cross_path_observations)} chars)\n\n"
                f"{synth.cross_path_observations}\n\n"
                f"### Conclusion ({len(synth.conclusion)} chars)\n\n"
                f"{synth.conclusion}"
            )

        # Round 1 各視点の素材
        for r in path_results:
            sections[f"(デバッグ) Round 1 視点別素材: {r.perspective_id}"] = (
                f"**視点**: {r.perspective_label}\n\n"
                f"**Section length**: {len(r.section_markdown)} chars\n\n"
                f"**Sources**: {len(r.sources)}\n\n"
                f"---\n\n"
                f"{r.section_markdown}\n\n"
                f"---\n\n"
                f"**Sources**:\n\n"
                + "\n".join(f"- [{s.title}]({s.url})" for s in r.sources)
            )

        # Round 2 各視点の素材 (peer_review enabled かつ Round 2 完走時のみ)
        if peer_review_enabled and path_results_revised:
            for r in path_results_revised:
                sections[
                    f"(デバッグ) Round 2 視点別素材: {r.perspective_id}"
                ] = (
                    f"**視点**: {r.perspective_label}\n\n"
                    f"**Section length**: {len(r.section_markdown)} chars\n\n"
                    f"**Sources**: {len(r.sources)}\n\n"
                    f"**Referenced siblings**: {r.referenced_siblings}\n\n"
                    f"**Revision notes**: {r.revision_notes}\n\n"
                    f"---\n\n"
                    f"{r.section_markdown}\n\n"
                    f"---\n\n"
                    f"**Sources**:\n\n"
                    + "\n".join(f"- [{s.title}]({s.url})" for s in r.sources)
                )

            # Round 1 vs Round 2 diff サマリ
            diff_lines = []
            for p in perspectives:
                r1 = r1_by_id.get(p.id)
                r2 = r2_by_id.get(p.id)
                if r1 is None or r2 is None:
                    continue
                r1_len = len(r1.section_markdown)
                r2_len = len(r2.section_markdown)
                delta = r2_len - r1_len
                changed = (
                    r2.revision_notes.strip()
                    != "significant change not required"
                )
                diff_lines.append(
                    f"### `{p.id}` — {p.label}\n\n"
                    f"- Round 1: {r1_len} chars / {len(r1.sources)} sources\n"
                    f"- Round 2: {r2_len} chars ({'+' if delta >= 0 else ''}{delta}) "
                    f"/ {len(r2.sources)} sources\n"
                    f"- Changed: {'yes' if changed else 'no (significant change not required)'}\n"
                    f"- Referenced siblings: {r2.referenced_siblings}\n"
                    f"- Revision notes: {r2.revision_notes}\n"
                )
            sections["(デバッグ) Round 1 vs Round 2 diff サマリ"] = (
                "\n".join(diff_lines) if diff_lines else "(no Round 2 results)"
            )

    saved_path = save_output(
        llm_name=args.llm,
        model_name=model_name,
        task=task,
        sections=sections,
    )
    print(f"  Saved to:         {saved_path}")


if __name__ == "__main__":
    main()
