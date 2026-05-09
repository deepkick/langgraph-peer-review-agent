---
pattern: peer_review
llm: anthropic
model: claude-sonnet-4-6
timestamp: '2026-05-09T10:30:09.499903'
task: GraphCast 登場以降の AI 気象予報モデルの技術系譜と、ECMWF AIFS / NOAA GraphCastGFS / Google WeatherNext
  の最新動向について整理してください。
---

## 最終レポート

# GraphCast 登場以降の AI 気象予報モデルの技術系譜と、ECMWF AIFS / NOAA GraphCastGFS / Google WeatherNext の最新動向について整理してください。

## はじめに

本レポートは「GraphCast 以降の AI 気象予報モデルの技術系譜と ECMWF AIFS・NOAA GraphCastGFS・Google WeatherNext の最新動向」という目標を、4 つの視点から多角的に照射する。**AI/ML 技術観点**はアーキテクチャの進化と手法的革新を、**データ・観測インフラ観点**は学習データ基盤とツールチェーンを、**歴史・発展経緯観点**は NWP からの転換と国際標準化の流れを、**国際比較・機関別比較観点**は主要 3 機関の戦略差異を担う。技術・データ・歴史・比較という 4 軸を組み合わせることで、単一視点では見えない「なぜ今この技術が実運用に至ったか」という全体像が浮かび上がる。

---

## AI/ML 技術観点：GraphCast 以降の全球気象 AI モデル系譜

### 1. GNN 起点と技術分岐 (2023–2024)

2023 年、Google DeepMind の **GraphCast** (Lam et al., *Science* 2023) が GNN ベース全球予報モデルとして登場し、質的転換をもたらした。0.25° 格子・227 変数を 6 時間ステップで自己回帰予測し、ECMWF HRES を 2,760 ターゲット中 89.3% で上回った。TPU 上で 10 日予報を 60 秒以内に生成できる推論速度も特筆される。

2024 年には Google DeepMind の **GenCast** (*Nature* 2024) が拡散モデルベースのアンサンブル手法として登場。15 日先まで予報し、ECMWF 51 メンバー ENS を 97.2% の検証指標で上回るなど、極端気象リスクの確率推定で高精度を示した。同年 ECMWF は **AIFS Single** (決定論的) を実運用化した。

### 2. AIFS-ENS の実運用とアーキテクチャ (2025)

ECMWF は 2025 年 7 月に **AIFS-ENS** を運用化した。アーキテクチャは「エンコーダ–プロセッサ–デコーダ」構造で、エンコーダ/デコーダに Transformer ベース GNN、プロセッサにスライディングアテンション窓付き Transformer を採用。パラメータ数 2.29 億、空間解像度約 30 km。ERA5 38 年分 + IFS 運用解析 8 年分で学習し、プロセッサ内ノイズ注入でアンサンブルメンバーを生成する。

学習損失には **CRPS (Continuous Ranked Probability Score)** を採用。拡散モデルより計算効率が高く精度も優れることを実証し、この手法は NVIDIA FourCastNet 3 などにも波及した。

同年、NOAA は GraphCast を基盤に自機関の GFS 解析データでファインチューニングした **AIGFS**（旧称 GraphCastGFS）を正式運用化した。熱帯低気圧トラック予報の改善と計算コストの大幅削減を実現した一方、v1.0 時点では強度予報に課題が残ることも公式に明記されている。データ配信の詳細はデータ・観測インフラ観点を参照されたい。

### 3. WeatherNext 2 と Anemoi フレームワーク (2025–2026)

2025 年 11 月、Google DeepMind は **WeatherNext 2** を発表した。中核技術は **Functional Generative Network (FGN)** と呼ぶ新アーキテクチャで、モデル内部にノイズを直接注入することで物理的整合性を保ちながら確率的予報を生成する。単一 TPU で数百シナリオを 1 分以内に生成でき、従来比 8 倍の高速化を達成。前世代 WeatherNext を 99.9% の変数・リードタイムで上回り、Earth Engine / BigQuery / Vertex AI 経由で提供されている。

2026 年 5 月、ECMWF は IFS Cycle 50r1 と同時に **AIFS ENS v2** を実装した。主な変更点はマルチスケール損失の導入、デコーダのエッジ増強、物理整合性向上のための変数境界制約など。実装基盤には **Anemoi フレームワーク** (Apache-2.0) が採用され、外部コミュニティが AIFS を自前で実行できるオープン環境が整備された。国際比較・機関別比較観点が示すように、DWD の AICON など各国気象機関が Anemoi 上でモデルを構築する欧州協調体制が進展している。同時に実験的 ML モデル (Aurora / FourCastNet / GraphCast / Pangu-Weather) の並列運用は終了し、AIFS が唯一の AI 予報システムとして一本化された。

## データ・観測インフラ観点：気象 AI を支えるデータ基盤の現在地

### ERA5 ― 学習データの事実上の標準

ECMWF の全球再解析データセット **ERA5**（1979 年〜現在、0.25° 格子）は、GraphCast 以降のほぼすべての主要 AI 気象モデルの学習基盤となっている。AIFS の学習構成の詳細は AI/ML 技術観点を参照されたいが、データ基盤の観点では ERA5 の「学習データ標準」としての地位が 2025〜2026 年にさらに強固になった点が重要である。

2025 年秋、ECMWF は **Anemoi training-ready ERA5** を一般公開した。1979〜2023 年をカバーし、約 0.5 TB・65,000 ファイル超を **Zarr 形式**（クラウド最適化）で提供。解像度は約 **1 度**（ERA5 本体の 0.25° より粗いが、学習効率を重視した設計）で、気圧面変数 6 種×13 層＋単一面変数 23 種を収録し、**CC-BY-4.0** ライセンスで誰でも利用可能。「自前でデータセットを構築する」という最大の参入障壁を取り除いた点で、FAIR 原則の実践として画期的である。2026 年には Anemoi が欧州 AI 気象の共通データ基盤として正式に位置づけられ、DWD（ドイツ気象局）など各国気象機関が Anemoi datasets パッケージ（GRIB/NetCDF → Zarr 変換）を通じて独自の学習データセットを構築する体制が整いつつある（国際比較・機関別比較観点も参照）。

### オープンデータ配信とフォーマット標準

ECMWF Open Data（AWS レプリカ含む）は AIFS リアルタイム予報を **GRIB2** 形式で配信する。NOAA の GFS データは AWS Open Data Registry 経由で公開され、GraphCastGFS（現 AIGFS）の初期値として活用される。民間事業者は GRIB2 → Zarr 変換パイプラインを AWS 上に構築し、新データ到着と同時に自動処理する運用を実現している。NOAA は **Cloud Optimized Zarr Reference Files（Kerchunk）** も整備し、オブジェクトストレージからの遅延読み込みを標準化しつつある。観測データ層では GOES・Meteosat・ひまわりの静止衛星 L1b データも AWS 上で公開されており、衛星観測を直接 AI 学習に組み込む研究が進む。

### ツールチェーンと評価基盤

Python エコシステムでは **xarray + cfgrib**（GRIB2 解析）、**Herbie**（AIFS/GFS/GraphCastGFS の統一取得）、**ecmwf-opendata**（公式クライアント）が標準的なツールチェーンを形成する。評価基準としては **WeatherBench2**（Google、2023〜）が事実上の共通ベンチマークとして機能し、ERA5 を真値として各モデルの RMSE・ACC を比較可能にしている。AI モデル設計の詳細は AI/ML 技術観点を参照されたい。

## 歴史・発展経緯：数値予報からAI実運用常態化へ

### NWP の限界と黎明期 (〜2022年)

数値天気予報 (NWP) は 1950 年の Charney らによる ENIAC 実験を起点とし、Lorenz の 1960 年代のカオス理論が予測可能性の上限を規定した。以降 70 年間、物理方程式の数値積分が気象予報の基盤であり続けた。2018〜2022 年に機械学習の気象応用が試みられたが、精度・汎化性ともに NWP に及ばず補助的な位置づけにとどまった。この時期の手法は「AI 文脈への接続点」として位置づけられる。

### 2023年: GraphCast による質的転換

2023 年、Google DeepMind が Science 誌に GraphCast (Lam et al., doi:10.1126/science.adi2336) を発表。GNN ベースの全球予報モデルが ECMWF HRES を 2,760 指標中 89.3% で上回り、単一 TPU で 1 分以内に 10 日予報を生成できることを示した。同年 7 月には Huawei Pangu-Weather が Nature 誌に掲載され、AI モデルが NWP と対等以上の精度を持つことが査読論文で相次いで実証された。この「2023 年の質的転換」が実運用移行への扉を開いた。アーキテクチャの技術詳細は AI/ML 技術観点を参照。

### 2024〜2026年: 実運用元年から国際標準化へ

2024 年に ECMWF が AIFS Single (決定論的モデル) を世界初の主要気象機関による AI モデル運用化として開始。2025 年初頭には IFS との並列稼働体制へ移行し、同年 7 月にアンサンブル版 AIFS-ENS を展開した。2026 年 5 月には AIFS ENS v2 が IFS Cycle 50r1 と同時実装され、実験的並列運用モデル群が整理されて AIFS が唯一の AI 予報システムとして一本化された (詳細は AI/ML 技術観点を参照)。

米国では NOAA が 2025 年 12 月に AIGFS (Artificial Intelligence Global Forecast System) および AIGEFS・HGEFS を正式運用化した。GraphCastGFS を研究段階 (Project EAGLE) で発展させ、GFS 解析値で追加学習した AIGFS v1.0 は熱帯低気圧トラック予報の改善と計算コスト削減を実証した一方、強度予報は今後の課題として明記されている。

国際的枠組みとして WMO は 2025 年 9 月にアブダビで「AI for Weather Prediction」会議を開催し、「AI は世界の早期警報・意思決定システムに組み込まれるべき」とする会議声明を採択。この声明は同年 10 月の WMO 臨時会議への提言として機能し、AI を気象予報インフラの不可欠な要素として国際的に位置づける議論を主導した。WMO 副事務局長は「AI はまだ局所的高影響イベントに限界があり、大規模展開前に解決が必要」とも明言しており、楽観と慎重のバランスが国際合意の特徴となっている。

2026 年現在、ECMWF AIFS・NOAA AIGFS・Google WeatherNext 2 が並列稼働する「AI 実運用の常態化」フェーズに入り、各機関の競争と協調が気象予報の新標準を形成しつつある。データ基盤の整備状況はデータ・観測インフラ観点、各機関の戦略比較は国際比較・機関別比較観点を参照されたい。

## 国際比較・機関別比較観点：AI 気象予報を主導する機関の戦略と運用状況

### ECMWF — 欧州協調型・オープン化の旗手

ECMWF は AI 気象予報の制度的中心として段階的な実運用化を達成した。**AIFS Single** は 2025 年初頭に正式運用を開始し、初日から世界 46 拠点へ 130 GB の予報データを配信。続いて **AIFS-ENS が 2025 年 7 月に運用化** され、確率的アンサンブル予報も AI で提供する体制が整った。両モデル合計で日次約 6 TB のデータを生成し、全量を Open Data として無償公開している。

2026 年 5 月には **AIFS ENS v2** を IFS Cycle 50r1 と同時実装し、AI/ML 技術観点が詳述するように実験的並列運用モデル群（Aurora / FourCastNet / GraphCast / Pangu-Weather）を終了して AIFS を唯一の AI 予報システムとして一本化した。**Anemoi フレームワーク**（Apache-2.0）を欧州 AI 気象の共通基盤として正式位置づけ、ドイツ気象局（DWD）の AICON など各国気象機関が Anemoi 上でモデルを構築・運用する「欧州協調体制」が実現しつつある。EU の Destination Earth イニシアティブとも連携し、オープンソース設計が透明性と再現性を担保する戦略的柱となっている。

### NOAA — 米国実験運用と産学連携

NOAA は 2025 年 12 月に **AIGFS・AIGEFS・HGEFS** の 3 モデルを同時に正式運用化した。AIGFS（決定論的 AI 全球予報）は Google DeepMind の GraphCast を基盤に GFS 解析データで追加学習したモデルで、EAGLE SOLO の後継にあたる。AIGEFS は 31 メンバーの AI アンサンブル（EAGLE Ensemble の後継）、HGEFS は物理ベース GEFS 31 メンバーと AIGEFS 31 メンバーを組み合わせた 62 メンバーの「グランドアンサンブル」であり、AI と物理モデルのハイブリッド運用という独自戦略を示す。熱帯低気圧トラック予報の改善と計算コスト削減を強調する一方、強度予報は v1.0 時点で課題が残るとも明記しており、段階的改善を公約している。Project EAGLE と Earth Prediction Innovation Center（EPIC）を通じた産学連携が開発加速の鍵で、AWS Public Data 経由のオープンデータ配信も継続している。

### Google DeepMind — 研究主導・クラウド実装

Google は 2025 年 11 月に **WeatherNext 2** を発表。Functional Generative Network（FGN）と呼ぶ新アーキテクチャにより、1 分以内に数百シナリオの確率的予報を生成でき、前世代モデルを 99.9% の変数・リードタイム（0〜15 日）で上回る。Earth Engine・BigQuery・Vertex AI（早期アクセス）での提供を開始し、研究から商用クラウドへの橋渡しを進める。さらに Google Search・Gemini・Pixel Weather・Maps Platform Weather API への統合も実施しており、気象 AI を消費者向けサービスに直接組み込む戦略は他機関にはない独自路線である。先行する GenCast（*Nature* 2024）が ECMWF 51 メンバー ENS を 97.2% の検証指標で上回ったことも、技術的優位性の根拠として示されている。

### 機関横断の比較と国際協調

3 機関の戦略を横並びで見ると、**ECMWF は欧州協調・オープン化**、**NOAA は AI×物理ハイブリッド**、**Google は研究→クラウド→消費者サービス** という異なる軸で展開していることが際立つ。WMO は 2025 年に AI を世界の早期警報インフラの一部として正式位置づける声明を採択し（歴史・発展経緯観点参照）、国際標準化の枠組みが整いつつある。NASA・JAXA は衛星観測データの AI 学習への直接組み込みで補完的役割を担い、データ・観測インフラ観点が示すように静止衛星 L1b データの AWS 公開がその基盤を支える。技術アーキテクチャの詳細は AI/ML 技術観点を、データ基盤はデータ・観測インフラ観点を参照されたい。

---

## 視点間の横断的観察

**一致点**：AI/ML 技術観点・データ・観測インフラ観点・歴史・発展経緯観点の 3 視点が、ERA5 を AI 気象モデルの学習データ標準として一致して強調している。また、2025 年 7 月の AIFS-ENS 運用化と 2026 年 5 月の AIFS ENS v2 一本化は、AI/ML 技術観点・歴史・発展経緯観点・国際比較・機関別比較観点の 3 視点すべてが「実運用常態化の象徴」として共通して位置づけている。

**相補点**：AI/ML 技術観点が AIFS-ENS の CRPS 損失やノイズ注入アーキテクチャを詳述する一方、データ・観測インフラ観点は Anemoi training-ready ERA5（Zarr 形式・CC-BY-4.0）の公開がその学習を可能にした基盤として補完する。国際比較・機関別比較観点が示す NOAA の「AI×物理ハイブリッド（HGEFS）」戦略は、歴史・発展経緯観点が描く「NWP との共存期」という文脈と重なり、段階的移行の必然性を裏付ける。

**緊張点**：国際比較・機関別比較観点は Google の消費者サービス統合という独自路線を肯定的に描くが、歴史・発展経緯観点が引用する WMO 副事務局長の「局所的高影響イベントへの限界」という慎重論は、商用展開の先行に対する制度的留保として対置される。

## 結論

GraphCast（2023）を起点とする AI 気象予報は、2026 年現在「実運用の常態化」フェーズに到達した。ECMWF はオープン協調、NOAA は AI×物理ハイブリッド、Google は研究からクラウド・消費者サービスへという三者三様の戦略が並走する。ERA5 と Anemoi フレームワークがデータ・ツール両面の共通基盤を形成し、WMO の国際声明が制度的正統性を付与した。一方で強度予報や局所的高影響イベントへの限界は未解決であり、技術競争と国際協調の両輪が次の標準を形成する段階にある。

---

## 出典

- [GraphCast: Learning skillful medium-range global weather forecasting](https://www.science.org/doi/10.1126/science.adi2336) (引用: `ai_tech`)
- [GenCast: Diffusion-based ensemble weather forecasting at scale](https://www.nature.com/articles/s41586-024-08252-9) (引用: `ai_tech`)
- [WeatherNext 2: Our most advanced weather forecasting model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/) (引用: `ai_tech`)
- [WeatherNext 2 - Google DeepMind](https://deepmind.google/science/weathernext/) (引用: `ai_tech`)
- [NOAA deploys new generation of AI-driven global weather models](https://www.noaa.gov/news-release/noaa-deploys-new-generation-of-ai-driven-global-weather-models) (引用: `ai_tech`)
- [ECMWF AIFS Documentation](https://www.ecmwf.int/en/forecasts/documentation-and-support/aifs) (引用: `ai_tech`)
- [Introducing the Anemoi training-ready version of ERA5 | ECMWF AIFS Blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2025/introducing-anemoi-training-ready-version-era5) (引用: `data_infra`)
- [ECMWF Newsletter No. 185 – Autumn 2025](https://www.ecmwf.int/sites/default/files/elibrary/102025/81689-newsletter-no-185-autumn-2025.pdf) (引用: `data_infra`)
- [Anemoi: a European framework for operational AI in weather and climate | ECMWF (2026)](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/anemoi-european-framework-ai) (引用: `data_infra`)
- [ECMWF Newsletter No. 186 – Winter 2025/26](https://www.ecmwf.int/sites/default/files/elibrary/012026/81713-newsletter-no-186-winter-202526.pdf) (引用: `data_infra`)
- [Conference charts way forward on potential for AI in weather prediction](https://wmo.int/media/news/conference-charts-way-forward-potential-ai-weather-prediction) (引用: `history_evolution`)
- [Conference Statement AI for Weather Prediction (PDF)](https://wmo.int/sites/default/files/2025-09/Conference%20Statement%20180925.pdf) (引用: `history_evolution`)
- [NOAA EAGLE (Experimental AI Global and Limited-Area Ensemble) — AWS Open Data Registry](https://registry.opendata.aws/noaa-nws-graphcastgfs-pds) (引用: `history_evolution`)
- [NOAA EAGLE Global Deterministic and Ensemble Forecasts — Registry of Open Data on AWS](https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/) (引用: `comparative_intl`)
- [WeatherNext 2: Google DeepMind's most advanced forecasting model](https://deepmind.google/blog/weathernext-2-our-most-advanced-weather-forecasting-model/) (引用: `comparative_intl`)
- [WeatherNext 2 — Google DeepMind Science](https://deepmind.google/science/weathernext) (引用: `comparative_intl`)
- [WeatherNext 2: Our most advanced weather forecasting model (blog.google)](https://blog.google/technology/google-deepmind/weathernext-2/) (引用: `comparative_intl`)

