"""Replay capture & playback support for multi_path_peer_review."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .perspectives import PathResult, Perspective
from .models import SynthesizerOutput

REPLAY_SCHEMA_VERSION = "1.0"


class ReplayRecorder:
    """LangGraph stream の (node_name → updates) emit を JSON stage に変換して蓄積する。

    使い方:
        recorder = ReplayRecorder(metadata={...})
        recorder.start()
        for mode, payload in graph.stream(initial, stream_mode=["updates", "values"]):
            if mode == "updates":
                recorder.on_update(payload)
            elif mode == "values":
                final_state = payload  # 最新の state を保持
        recorder.finalize(final_state)
        recorder.save(output_path)
    """

    def __init__(self, metadata: dict[str, Any]):
        # metadata は外から注入: id, title, description, task, llm, model, peer_review_enabled
        self.metadata = dict(metadata)
        self.stages: list[dict[str, Any]] = []
        self.start_time: datetime | None = None

    def start(self) -> None:
        self.start_time = datetime.now()
        self.metadata["started_at"] = self.start_time.isoformat()

    def on_update(self, update: dict) -> None:
        """LangGraph の (node_name → updates) emit 1 件を処理する。"""
        if self.start_time is None:
            raise RuntimeError("start() を先に呼んでください")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        for node_name, updates in update.items():
            stage = self._classify_stage(node_name, updates, elapsed)
            if stage is not None:
                self.stages.append(stage)

    def _classify_stage(
        self, node_name: str, updates: dict, elapsed: float
    ) -> dict | None:
        """node_name と updates の中身から stage 辞書を構築する。"""
        if node_name == "selector":
            perspectives = updates.get("perspectives", [])
            return {
                "stage": "selector_done",
                "elapsed_seconds": elapsed,
                "perspectives": [p.model_dump() if hasattr(p, "model_dump") else p
                                 for p in perspectives],
                "selector_rationale": updates.get("selector_rationale", ""),
            }

        if node_name == "path_executor":
            # Annotated[list, operator.add] reducer なので、updates["path_results"] は
            # 1 件だけの increment list (この invocation で生成された PathResult)
            results = updates.get("path_results", [])
            if not results:
                return None
            r = results[0]
            return {
                "stage": "round1_path_done",
                "elapsed_seconds": elapsed,
                "perspective_id": r.perspective_id,
                "perspective_label": r.perspective_label,
                "section_markdown": r.section_markdown,
                "sources": [s.model_dump() for s in r.sources],
            }

        if node_name == "path_executor_round2":
            results = updates.get("path_results_revised", [])
            if not results:
                return None
            r = results[0]
            return {
                "stage": "round2_path_done",
                "elapsed_seconds": elapsed,
                "perspective_id": r.perspective_id,
                "perspective_label": r.perspective_label,
                "section_markdown": r.section_markdown,
                "sources": [s.model_dump() for s in r.sources],
                "referenced_siblings": r.referenced_siblings,
                "revision_notes": r.revision_notes,
            }

        if node_name == "round2_router":
            # no-op gate なのでスキップ
            return None

        if node_name == "synthesizer":
            synth: SynthesizerOutput | None = updates.get("synthesizer_output")
            if synth is None:
                return None
            return {
                "stage": "synthesizer_done",
                "elapsed_seconds": elapsed,
                "introduction": synth.introduction,
                "cross_path_observations": synth.cross_path_observations,
                "conclusion": synth.conclusion,
            }

        if node_name == "assembly":
            return {
                "stage": "assembly_done",
                "elapsed_seconds": elapsed,
                "final_report": updates.get("final_report", ""),
            }

        return None

    def finalize(self, final_state: dict) -> None:
        """run 完了時にメタデータを更新する。"""
        if self.start_time is None:
            raise RuntimeError("start() を先に呼んでください")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.metadata["total_elapsed_seconds"] = elapsed
        self.metadata["final_report_length"] = len(final_state.get("final_report", ""))
        perspectives = final_state.get("perspectives", [])
        self.metadata["perspectives_selected"] = [p.id for p in perspectives]
        self.metadata["perspectives_selected_labels"] = [p.label for p in perspectives]
        self.metadata["schema_version"] = REPLAY_SCHEMA_VERSION

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "stages": self.stages,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
