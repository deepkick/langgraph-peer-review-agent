# KNOWLEDGE.md

実装・運用を通じて articulate された学びを蓄積するドキュメント。設計原則は `docs/DESIGN.md` に、運用知見はこちらに記録する。

## Gradio 6.x API specifics

- `theme` parameter は `gr.Blocks()` ではなく `demo.launch()` に渡す。Gradio 6.0 で API が変更され、旧位置 (`gr.Blocks(theme=...)`) に置くと UserWarning が出る。
- `font` / `font_mono` は raw `str` ではなく `gr.themes.Font(name)` で wrap する必要がある。生 str を渡すと内部 equality check (`Font.__eq__`) が `'str' object has no attribute 'name'` で AttributeError を起こす。`gr.themes.LocalFont` は 6.14.0 に存在せず、`gr.themes.Font(name)` のみが利用可能。
- `gr.themes.Soft()` のデフォルトフォント (Quicksand) は丸みが強く、日本語混在文書での Latin 可読性が劣化する。システムフォント (`["ui-sans-serif", "system-ui", "-apple-system", "sans-serif"]`) にフォールバックする形で上書きするのが無難。

## HF Space deployment: dual-repo + sync script pattern

- 単一 repo + frontmatter 方式は GitHub README が見栄え regression する。GitHub は YAML frontmatter を hidden 扱いせず raw 表示するため、portfolio 用途の README が崩れる。
- 採用パターン: GitHub repo と HF Space repo を分け、HF 側を `~/Documents/AI_Agent/<project>-hf/` として sibling 配置する dual-repo 方式。`weather-ai-rag-advanced` と統一。
- 操作 cost を `scripts/sync_to_hf.sh` に集約。app.py と examples/replays/ のみを sync し、commit + push までを 1 コマンドで完結。
  ```
  ./scripts/sync_to_hf.sh "<commit message>"
  ```

## Safari basic auth 不具合

- Gradio の `auth=(user, pass)` 基本認証は Safari で動作不安定。
- 原因候補: Intelligent Tracking Prevention / Keychain auto-fill / 3rd-party cookie 制限。
- 対応: 勉強会案内に「Chrome / Firefox 推奨」を明記する。app.py 側の修正は不要 (Chrome / Firefox では問題なく動作)。

## Briefing pattern + Stop points

- Phase 着手前に詳細 briefing を `_briefings/<phase>.md` (gitignored) として作成し、Claude Code に渡して順次実行させる。
- Stop points を予め複数定義する。Phase 2.2 では 4 つ: コスト gate (Step 3 着手前) / 品質判断 (replay 視点選定確認) / バージョン確認 (Gradio install) / 手作業フェーズ (HF Spaces deploy)。
- Claude Code が定義済 Case A-I のいずれかに該当する判断を要する場面では停止して Kaoru に判断を仰ぐ。
- 効果: 誤実行抑制、人間の判断と Claude Code の自動化の役割分離。Phase 単位で履歴が `git log --first-parent` で読める。

## ReplayRecorder + run_streaming 設計

- `graph.stream(stream_mode=["updates", "values"])` の dual mode を採用。
  - `"updates"`: 各 node の delta (recorder が stage 分類に使用)
  - `"values"`: 完全な state スナップショット (final_state として常に最新で上書き)
- 手動 reduce を避けることで、State スキーマ変更時の保守コストを排除し、LangGraph の reducer 挙動を正本として final state を取得できる。
- recorder が `None` の場合は更新通知をスキップ (`run()` と同等動作)、CLI 互換性を維持。

## Project-local venv の意義

- 共有 venv の問題: 隠れた依存混入 ("works locally, fails on deploy")、バージョン競合、再現性の欠如。
- 対策: project-local `.venv/` を `uv venv --python 3.12 .venv` で作成し、`requirements.txt.lock` を `uv pip compile` で生成。
- `weather-ai-rag-advanced` と同パターン。Phase 2.3 で BYOK 等の新 deps を追加する際に、依存影響を本 project 内に閉じ込められる。

## HF CLI modernization

- `huggingface-cli` は deprecated、`hf` コマンドに移行。
- 主要対応:
  - `huggingface-cli login` → `hf auth login`
  - `huggingface-cli whoami` → `hf auth whoami`
- credential は `~/.cache/huggingface/token` (global、venv 非依存) に保存されるため、project-local venv 移行時に追加 install / 再認証は不要。
