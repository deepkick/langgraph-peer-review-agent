"""multi_path_peer_review の共有定義モジュール (型・定数・ドメイン文脈)。

このモジュールは、multi_path エージェントの全 sub-agent
(perspective_selector / path_executor / path_executor_round2 /
cross_path_synthesizer) で共有するデータ構造・定数・ドメイン文脈を一元管理する。

含まれるもの:
- 構造的パラメータ: NUM_PATHS, MAX_ITERATIONS_PER_PATH
- 視点定義: Perspective (BaseModel), PERSPECTIVE_POOL, PerspectiveID (Literal)
- selector 出力: SelectedPerspectives
- path 出力: PathSource, PathResult
- ドメイン文脈: DOMAIN_CONTEXT (Tier 1/2/3 情報源、時代背景、ツール標準)
- ヘルパー関数: format_perspective_pool_md, get_pool_summary, label_for_id

設計判断:
- Core 3 視点 (ai_tech, data_infra, genai_application) と補完 5 視点に分類
- 視点 ID は Pydantic Literal で型レベル制約 (制約の 4 段階モデル 段階 2)
- 4 視点固定 (NUM_PATHS = 4) を Pydantic min/max_length で型レベル強制
- DOMAIN_CONTEXT は selector / path_executor で共有して指示の整合性を保つ
- 重点情報源を Tier 構造で明示 (ECMWF / NOAA / DeepMind を絶対優先)
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ===== 構造的パラメータ =====


# 1 クエリあたり選定する視点数
NUM_PATHS = 4


# 1 パスあたりの Tavily 検索ループの最大反復数。
# 視点で絞られているため少ない値で十分。
MAX_ITERATIONS_PER_PATH = 2


# Tavily 検索結果の最低 score 閾値 (これ未満は自動除外)。
# 制約の 4 段階モデル 段階 4: コード側での品質保証。
#
# 0.5 採用の理由:
# - 0.6 は厳しすぎた: Tier 1 の中品質結果 (PDF 内ページや周辺記事) まで
#   巻き込んで除外する事象が頻発した
# - 0.5 なら全件除外を回避しつつ、明らかな低品質 (score < 0.5) は防げる
# - ブラックリスト方式は不採用 (LLM 自由度尊重):
#   * SEO サイトは無限に生まれ、維持コストが青天井
#   * 個人ブログ / Medium 記事は genai_application 視点で不可欠
#   * ソース品質の文脈判断は LLM の責務
MIN_TAVILY_SCORE = 0.5


# ===== 視点 ID の Literal 型定義 =====
# 制約の 4 段階モデル 段階 2: 構造化出力による型強制


PerspectiveID = Literal[
    # Core 3 視点
    "ai_tech",
    "data_infra",
    "genai_application",
    # 補完 5 視点
    "meteorology",
    "disaster_society",
    "policy_ethics",
    "history_evolution",
    "comparative_intl",
]


# ===== Perspective データ定義 =====


class Perspective(BaseModel):
    """1 つの視点の定義。curated list の要素。"""

    id: PerspectiveID
    label: str
    description: str


# ===== curated list =====
# Core / 補完の区別はラベルで明示。selector はこれを読んで選定する。


PERSPECTIVE_POOL: list[Perspective] = [
    # ---- Core 3 視点 ----
    Perspective(
        id="ai_tech",
        label="AI/ML 技術観点 [Core]",
        description=(
            "気象 × AI の中核となる技術観点。GNN / Transformer / Diffusion ベースの"
            "全球気象 AI モデルのアーキテクチャ、学習手法、推論性能、評価指標 (RMSE/ACC) を深掘りする。"
            "重点モデル: GraphCast (2023) を起点に、AIFS / AIFS-ENS (ECMWF 実運用)、"
            "GraphCastGFS (NOAA 研究用)、GenCast (アンサンブル拡散)、"
            "WeatherNext 2 (Google DeepMind)、Anemoi (AIFS 系列の LAM フレームワーク)。"
            "オープンソース・オープンウェイト (Apache-2.0 等) を優先。"
            "時間軸は 2023 年以降を中心に、2025-2026 年の最新動向を重視 "
            "(2022 年以前の手法は AI 文脈への接続点としてのみ扱う)。"
            "※物理プロセスの議論は meteorology、データ基盤は data_infra 観点で扱う。"
            "※LLM / 汎用生成 AI による解析・自動化は genai_application 観点で扱う。"
        ),
    ),
    Perspective(
        id="data_infra",
        label="データ・観測インフラ観点 [Core]",
        description=(
            "気象 AI を支えるデータ基盤とツールチェーンの観点。"
            "**ERA5 (ECMWF 全球再解析) が気象 AI の事実上の学習データ標準** である点を起点に、"
            "WeatherBench2 (評価用)、ECMWF Open Data (AIFS 配信)、"
            "AWS Public Data (NOAA GraphCastGFS) といったオープンデータ群を扱う。"
            "ツールチェーン: GRIB2 / NetCDF / Zarr (フォーマット)、xarray + cfgrib (解析標準)、"
            "Herbie (AIFS/GFS/GraphCastGFS 取得)、ecmwf-opendata (公式クライアント)、cartopy (可視化)。"
            "観測データ層: 気象衛星 (ひまわり、GOES、Meteosat)、地上観測 (アメダス)、"
            "Anemoi 的マルチスケール構造。"
            "FAIR 原則とオープン化の進展を重視。"
            "※AI モデル設計は ai_tech、運用配信は disaster_society 観点で扱う。"
        ),
    ),
    Perspective(
        id="genai_application",
        label="生成AI 活用観点 [Core]",
        description=(
            "気象 × AI 研究を加速する **汎用生成 AI / LLM の活用観点** (Core 3 視点の 1 つ)。"
            "気象専用予報 AI (ai_tech 観点) とは別軸で、研究加速・データ解析自動化・"
            "コミュニケーション支援を担う汎用生成 AI の役割を扱う。"
            "重点領域: "
            "(1) LLM (Claude / GPT / Gemini) による研究のコード生成・論文サマリ・仮説生成、"
            "(2) 気象データ解析エージェント (LangChain / LangGraph で構築する RAG・自動化フロー)、"
            "(3) 気象向け基盤モデル (Microsoft Aurora、NVIDIA FourCastNet、ClimaX 系)、"
            "(4) マルチモーダル LLM による衛星画像・気象チャート・予報マップの解釈、"
            "(5) WMO「AI声明」が言及する「責任ある AI 統合」の文脈。"
            "重点組織: Anthropic、OpenAI、Google、Microsoft Research、NVIDIA、"
            "LangChain (フレームワーク提供)。"
            "**気象専用 AI モデル (GraphCast/AIFS/GenCast 等) の技術詳細は ai_tech 観点で扱う**。"
            "境界線上にある気象基盤モデル (Aurora 等) は、"
            "汎用性・事前学習・下流タスク多様性の文脈ならこちら、"
            "専用予報モデルとしての文脈なら ai_tech へ。"
        ),
    ),
    # ---- 補完 5 視点 ----
    Perspective(
        id="meteorology",
        label="気象学・地球科学観点 (補完)",
        description=(
            "気象 AI が扱う対象 (大気・海洋・地表) の物理プロセスを解説する補完観点。"
            "AI モデルが何を予測しているのか・物理的にどこまで妥当かを評価する視点。"
            "トピック例: 数値予報の物理基礎、線状降水帯のメカニズム、"
            "極端現象 (台風・熱波・豪雨) の物理、"
            "データ駆動 AI と物理ベース NWP の整合性 (物理 IFS と AIFS の並列稼働等)。"
            "※AI モデルの内部構造・学習手法は ai_tech 観点で扱う。"
        ),
    ),
    Perspective(
        id="disaster_society",
        label="防災・社会実装観点 (補完)",
        description=(
            "気象 AI が現場・社会にどう届くかの補完観点。"
            "WMO の「Early Warnings for All」と 2025 年 10 月の「AI声明」"
            "(AI を世界の警報インフラの一部として位置づけ) を背景に、"
            "物理 NWP と AI 予報の並列稼働、AI 台風予報のメディア露出 (NHK 等)、"
            "自治体・住民への警報配信、防災・減災への実装を扱う。"
            "日本では気象庁の社会実装 (台風予報・線状降水帯・警報システム) も含む "
            "(AI 技術観点での気象庁は ai_tech では優先度が下がる点に注意)。"
            "※倫理・公平性・規制論点は policy_ethics 観点で扱う。"
        ),
    ),
    Perspective(
        id="policy_ethics",
        label="政策・倫理・ガバナンス観点 (補完)",
        description=(
            "気象 AI の社会的・制度的論点を扱う補完観点。"
            "「誰がルールを作り、誰が恩恵を受け、誰が責任を持つか」の視点。"
            "WMO 2025 年 10 月「AI声明」(World Meteorological Congress endorses AI for "
            "forecasts and warnings) を起点に、以下の 3 軸を扱う: "
            "(1) **制度・標準化**: 各国気象機関のオープンデータ方針、"
            "ライセンス体系 (Apache-2.0 / Open Data / FAIR 原則)、"
            "輸出規制 (例: Google WeatherNext の Restricted Country リスト、日本も含まれる)、"
            "官民学連携、"
            "(2) **倫理・公平性**: グローバルサウス / 島嶼国への AI 恩恵格差、"
            "学習データの地域偏り (ERA5 の偏在)、Early Warnings for All、"
            "(3) **責任・透明性**: 誤予測時の責任帰属 (ECMWF・NOAA・気象庁・民間気象会社)、"
            "ブラックボックス問題、説明可能性、説明責任。"
            "日本では気象業務法による気象庁特別扱いが AI オープン化の論点 "
            "(理解可能だが周回遅れ要因にもなっている)。"
        ),
    ),
    Perspective(
        id="history_evolution",
        label="歴史・発展経緯観点 (補完)",
        description=(
            "気象 AI の発展経緯を整理する補完観点。"
            "数値予報の歴史的背景 (Charney 1950s, Lorenz 1960s) を NWP 文脈の起点として、"
            "2018-2022 年の機械学習気象応用の黎明期、"
            "**2023 年の GraphCast 登場による質的転換**、"
            "2024-2025 年の実運用元年 (AIFS/GraphCastGFS/WeatherNext)、"
            "2025 年 WMO「AI声明」による国際運用段階移行、という流れを扱う。"
            "**2022 年以前の古典的内容は AI 文脈への接続点としてのみ扱う**。"
            "現在進行中の技術詳細は ai_tech 観点で扱う。"
        ),
    ),
    Perspective(
        id="comparative_intl",
        label="国際比較・機関別比較観点 (補完)",
        description=(
            "AI 気象を主導する機関を横並びで比較する補完観点。"
            "**重点機関**: "
            "ECMWF (AIFS / AIFS-ENS / Anemoi / Open Data の中心、欧州協調体制)、"
            "NOAA / NCEP (GraphCastGFS / AWS Public Data、米国実験運用)、"
            "Google DeepMind (GraphCast / GenCast / WeatherNext シリーズ)、"
            "NASA / JAXA (衛星観測 × AI 融合)、WMO (国際協調・標準化)。"
            "**機関ごとの戦略・モデル・運用ステータス・オープン化方針** を比較する。"
            "防災・社会実装観点では日本の気象庁も対象に含めるが、"
            "AI 技術リーダーシップ観点では欧米系が中心。"
            "※特定機関の単独深掘りは ai_tech や data_infra 観点で扱う。"
        ),
    ),
]


# ===== SelectedPerspectives (selector の出力型) =====


class SelectedPerspectives(BaseModel):
    """perspective_selector が curated list から選定した結果。

    制約の 4 段階モデル 適用:
    - 段階 2 (型強制): selected_ids は PerspectiveID Literal で 8 個に制限、
      min_length=max_length=NUM_PATHS で 4 個固定
    - 段階 2 (Pydantic validator): no_duplicates で重複検出
    - 段階 1 (プロンプト): selector のプロンプトで上記を再強調
    """

    selected_ids: list[PerspectiveID] = Field(
        ...,
        min_length=NUM_PATHS,
        max_length=NUM_PATHS,
        description=f"curated list から選んだ厳密に {NUM_PATHS} 個の視点 ID",
    )
    rationale: str = Field(
        ...,
        description=(
            "この 4 視点を選んだ理由。"
            "冒頭 1-2 文で全体方針、各視点について 1 文ずつ何を担当するかを記述。"
            "200-400 文字。"
        ),
    )

    @field_validator("selected_ids")
    @classmethod
    def no_duplicates(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"selected_ids に重複があります: {v}")
        return v


# ===== Path 出力型 (path_executor の出力 / cross_path_synthesizer の入力) =====


class PathSource(BaseModel):
    """1 つのパスが引用する出典 (Tavily 検索結果由来)。"""

    title: str = Field(..., description="出典タイトル")
    url: str = Field(..., description="出典 URL")
    snippet: str = Field(
        default="",
        description="関連性のある短い抜粋 (引用元の確認用、空でも可)",
    )


class PathResult(BaseModel):
    """1 つのパス (= 1 視点) の実行結果。

    cross_path_synthesizer はこれを集約して最終レポートを作成する。

    制約の 4 段階モデル 適用:
    - 段階 2 (型強制): perspective_id は PerspectiveID Literal で 8 個に制限
    - 段階 1 (プロンプト): section_markdown の文字数目標 (1,200-1,400) はプロンプトで指示

    Round 2 (peer-review) 用フィールド:
    - round, referenced_siblings, revision_notes の 3 フィールド (defaults あり)
    - Round 1 では空のまま

    section_markdown の max_length=2500 は Round 2 の改訂機会を保護するため
    (上流 _PathRound2LLMOutput.max_length=2500 と同期)。Round 1 は依然として
    上流の _PathLLMOutput.max_length=1700 で制限されているため、Round 1 の
    strict gate は機能している。
    """

    perspective_id: PerspectiveID = Field(..., description="担当した視点の ID")
    perspective_label: str = Field(..., description="担当した視点の label")
    section_markdown: str = Field(
        ...,
        max_length=2500,
        description=(
            "担当視点に基づくセクション本文。Markdown 形式、見出し ## から開始。"
            "Round 1 では 1,200-1,400 字以内 (上流の _PathLLMOutput.max_length=1700 で制限)。"
            "Round 2 は最小限の差分原則のもと Round 1 ± 200 字程度を目安。"
            "max_length=2500 は両 Round の最大値に揃えた構造的安全策 (異常時の防御)。"
            "視点の核心を凝縮し、サブセクションは 3 個以内に抑える。"
        ),
    )
    sources: list[PathSource] = Field(
        default_factory=list,
        description="このパスで引用した Tavily 検索結果のリスト",
    )

    # ===== Round 2 (peer-review) 用フィールド =====
    round: Literal[1, 2] = Field(
        default=1,
        description="この PathResult がどのラウンドの出力か",
    )
    referenced_siblings: list[PerspectiveID] = Field(
        default_factory=list,
        description=(
            "Round 2 で本文中に明示的に参照した他視点の id のリスト。"
            "Round 1 では空。"
        ),
    )
    revision_notes: str = Field(
        default="",
        description=(
            "Round 2 で何を変えたか / なぜ。1-3 文。"
            "改訂不要の場合は 'significant change not required' と書く。"
            "Round 1 では空。"
        ),
    )


# ===== ドメイン文脈 (selector / path_executor 両方で参照) =====
# サブエージェント間のプロンプト矛盾を予防するため、共有定数として一元管理する。


DOMAIN_CONTEXT = (
    "## 気象 × AI ドメイン前提\n\n"
    "### 時代背景\n"
    "- 2023 年: GraphCast 登場による質的転換 (GNN ベース全球予報モデル)\n"
    "- 2024-2025 年: 実運用フェーズ開始 — ECMWF AIFS / AIFS-ENS、"
    "NOAA GraphCastGFS、Google WeatherNext 2 が並列稼働\n"
    "- 2025 年 10 月: WMO「AI声明」採択 — AI を世界の警報インフラの一部として正式位置づけ\n"
    "- **2022 年以前の手法は AI 文脈への接続点としてのみ扱う。"
    "中心は GraphCast 以降と 2025-2026 年の最新動向**\n\n"
    "### 最重要情報源 (調査時はここを最優先)\n"
    "**Tier 1 — 主導 3 機関の公式発表・論文 (絶対優先)**:\n"
    "1. **ECMWF (欧州中期予報センター)** — 気象 × AI の中心\n"
    "   - 公式: ecmwf.int (media-centre, forecasts, AIFS blog)\n"
    "   - データ: ECMWF Open Data (AIFS), set-ix\n"
    "   - 関連: AIFS / AIFS-ENS / Anemoi の論文・technical memorandum\n"
    "2. **NOAA / NCEP (米国海洋大気庁)** — GraphCast 運用・データ配信\n"
    "   - 公式: noaa.gov, registry.opendata.aws/noaa-nws-graphcastgfs-pds\n"
    "3. **Google DeepMind** — GraphCast / GenCast / WeatherNext の発信元\n"
    "   - 公式: deepmind.google/science/weathernext, blog.google/technology\n"
    "   - 論文: Lam et al. 2023 (GraphCast, Science)、"
    "Price et al. (GenCast, Nature) など\n\n"
    "**Tier 2 — 重要機関**: NASA / JAXA (衛星観測 × AI)、WMO (国際協調・標準化)\n\n"
    "**Tier 3 — 文脈補完**: 日本気象庁 (jma.go.jp) は社会実装 "
    "(台風予報・線状降水帯・警報システム) では重要だが、"
    "AI 技術観点では Tier 1 機関に比して優先度低\n\n"
    "### 生成 AI 活用領域の主要情報源 (genai_application 視点)\n"
    "- **Microsoft Research** (Aurora、ClimaX 系の気象基盤モデル): research.microsoft.com\n"
    "- **NVIDIA** (Modulus、FourCastNet 系): developer.nvidia.com/modulus\n"
    "- **Anthropic / OpenAI / Google** (基盤 LLM の進展): "
    "anthropic.com/news、openai.com/blog、blog.google/technology\n"
    "- **LangChain / LangGraph** (エージェントフレームワーク): "
    "python.langchain.com、langchain-ai.github.io/langgraph\n"
    "- 気象 × LLM のオープンソースプロジェクト・arXiv 論文\n\n"
    "### データ・ツール標準\n"
    "- 学習データ標準: **ERA5 (ECMWF 全球再解析)** + WeatherBench2 (評価)\n"
    "- フォーマット: GRIB2 / NetCDF / Zarr\n"
    "- 解析ツール: Python + xarray + cfgrib + cartopy + Herbie + ecmwf-opendata\n"
    "- ライセンス: Apache-2.0 / Open Data / FAIR 原則を重視\n\n"
    "### 重点モデル系譜 (2023 年以降)\n"
    "GraphCast (2023, GNN) → AIFS / AIFS-ENS (ECMWF 実運用) → "
    "GraphCastGFS (NOAA 研究) → GenCast (アンサンブル拡散) → "
    "WeatherNext 2 (2025) → Anemoi (LAM フレームワーク)\n\n"
    "### Tavily 検索ガイドライン (品質管理)\n"
    "**include_domains に推奨するドメイン**:\n"
    "- 気象 × AI 共通: ecmwf.int, noaa.gov, deepmind.google, blog.google, "
    "registry.opendata.aws, wmo.int, jaxa.jp, nasa.gov\n"
    "- 査読済み論文: nature.com, science.org\n"
    "- genai_application 視点で追加可: anthropic.com, openai.com, "
    "research.microsoft.com, developer.nvidia.com, python.langchain.com, "
    "langchain-ai.github.io, github.com\n\n"
    "**arxiv.org は include_domains に含めない**:\n"
    "- preprint は品質ばらつきが大きく、Tavily score だけでは識別困難\n"
    "- 既知の主要論文を引用したい場合は、本文中で arxiv ID を直接記述する形で対応\n"
    "  (例: GraphCast → arxiv:2212.12794、GenCast → Nature 2024 論文)\n\n"
    "**Tavily の挙動について (実証観察)**:\n"
    "- `include_domains` は **hint であり hard filter ではない**。"
    "指定外ドメインの結果も返ることがある (Tavily が「関連度が高い」と判断した場合)\n"
    "- `score` は **クエリとの関連度** であって、ソースの信頼性ではない。"
    "SEO 最適化されたサイトは高 score (0.99+) を取りやすい\n"
    "- コード側のフィルタは `score < 0.5` の自動除外のみ "
    "(機械的に判定可能な部分のみ防御)\n"
    "- **引用判断は score ではなく「発信元の権威性」で行う**:"
    "Tier 1 機関 (ECMWF / NOAA / DeepMind / WMO 等) > 査読論文 > 公式ブログ "
    "> 個人技術ブログ > 集約系・SEO サイト"
)


# ===== ヘルパー関数 =====


def format_perspective_pool_md(pool: list[Perspective] = PERSPECTIVE_POOL) -> str:
    """curated list を selector プロンプトに埋め込むための Markdown 整形。"""
    lines = []
    for p in pool:
        lines.append(f"### `{p.id}` — {p.label}")
        lines.append(p.description)
        lines.append("")
    return "\n".join(lines)


def get_pool_summary() -> str:
    """Pool の概要 (Run Configuration 表示用)。"""
    core_count = sum(1 for p in PERSPECTIVE_POOL if "[Core]" in p.label)
    supplementary_count = len(PERSPECTIVE_POOL) - core_count
    return (
        f"{len(PERSPECTIVE_POOL)} perspectives "
        f"({core_count} Core + {supplementary_count} 補完, select {NUM_PATHS})"
    )


def label_for_id(pid: str) -> str:
    """ID から表示用 label を返す。"""
    for p in PERSPECTIVE_POOL:
        if p.id == pid:
            return p.label
    return f"<unknown: {pid}>"
