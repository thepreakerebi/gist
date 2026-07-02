import argparse
import json
import re
import shutil
import shlex
import subprocess
import sys
from collections import Counter
from html import escape
from math import ceil
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.audio.whisper import TranscriptQuality
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
from gist.eval.quality import (
    QualityCaseDraft,
    QualityCase,
    QualityReport,
    draft_quality_case,
    load_quality_cases,
    render_quality_markdown,
    run_quality_cases,
)
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline
from gist.reports import render_local_compression_report

REQUIRED_QUERY_CATEGORIES = (
    QueryIntent.SPEECH_SEMANTIC,
    QueryIntent.VISUAL_OBJECT_ACTION,
    QueryIntent.TEMPORAL_BEFORE_AFTER,
    QueryIntent.GLOBAL_SUMMARY,
    QueryIntent.MIXED_AV,
)

_PROPOSAL_STOPWORDS = {
    "about",
    "actually",
    "after",
    "also",
    "because",
    "before",
    "case",
    "does",
    "from",
    "have",
    "into",
    "like",
    "most",
    "relevant",
    "say",
    "shown",
    "speaker",
    "that",
    "their",
    "there",
    "this",
    "what",
    "while",
    "with",
    "would",
}


class LongVideoSuiteGates(BaseModel):
    min_cases: Annotated[int, Field(ge=1)] = 30
    min_distinct_videos: Annotated[int, Field(ge=1)] = 5
    min_distinct_domains: Annotated[int, Field(ge=1)] = 3
    min_cases_per_category: Annotated[int, Field(ge=1)] = 3
    minimum_duration_seconds: Annotated[float, Field(gt=0)] = 3600.0
    min_quality_pass_rate: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_avg_token_reduction_percent: Annotated[float, Field(ge=0, le=100)] = 90.0
    max_noisy_transcript_warning_rate: Annotated[float, Field(ge=0, le=1)] = 0.25
    min_transcript_metadata_rate: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_answered_rate: Annotated[float, Field(ge=0, le=1)] = 0.9
    max_avg_selected_evidence: Annotated[float, Field(gt=0)] = 8.0


class LongVideoSuiteGateResult(BaseModel):
    name: str
    passed: bool
    actual: float
    required: float
    message: str


class LongVideoHealthSummary(BaseModel):
    artifacts: int = 0
    avg_token_reduction_percent: float = 0.0
    noisy_transcript_warning_rate: float = 0.0
    transcript_metadata_rate: float = 0.0
    answered_rate: float = 0.0
    avg_selected_evidence: float = 0.0
    quality_warning_counts: dict[str, int] = Field(default_factory=dict)
    transcript_quality_counts: dict[str, int] = Field(default_factory=dict)


class LongVideoExpansionPlan(BaseModel):
    needed_cases: int = 0
    needed_long_video_cases: int = 0
    needed_distinct_videos: int = 0
    needed_distinct_domains: int = 0
    needed_by_category: dict[str, int] = Field(default_factory=dict)
    priority_actions: list[str] = Field(default_factory=list)
    query_proposals: list["LongVideoQueryProposal"] = Field(default_factory=list)


class LongVideoQueryProposal(BaseModel):
    video_id: str
    domain: str
    query_category: QueryIntent
    query: str
    rationale: str
    source_artifact: Path


class LongVideoCurationResult(BaseModel):
    proposal: LongVideoQueryProposal
    video_path: Path
    compression_path: Path
    html_report_path: Path
    draft_case_path: Path
    review_json_path: Path
    review_markdown_path: Path
    draft: QualityCaseDraft


class LongVideoCurationReview(BaseModel):
    ready_for_dataset: bool = False
    warnings: list[str] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    recommendation: str

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoDraftAppendResult(BaseModel):
    review: LongVideoCurationReview
    appended: bool = False
    dataset_path: Path
    case_id: str | None = None


class LongVideoCurationQueueItem(BaseModel):
    proposal_index: int
    video_id: str
    domain: str
    query_category: QueryIntent
    query: str
    rationale: str
    source_artifact: Path
    command: str


class LongVideoCurationQueueReport(BaseModel):
    passed: bool
    case_count: int
    target_case_count: int
    needed_cases: int
    long_video_case_count: int
    needed_long_video_cases: int
    distinct_videos: int
    target_distinct_videos: int
    distinct_domains: int
    target_distinct_domains: int
    category_counts: dict[str, int]
    needed_by_category: dict[str, int]
    priority_actions: list[str]
    items: list[LongVideoCurationQueueItem]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoMetadataRefreshQueueItem(BaseModel):
    case_id: str
    video_id: str
    query: str
    compression_path: Path
    source_path: Path
    current_transcript_quality: str | None = None
    command: str
    command_args: list[str]


class LongVideoMetadataRefreshQueueReport(BaseModel):
    cases: int
    refresh_needed: int
    target_transcript_quality: TranscriptQuality
    items: list[LongVideoMetadataRefreshQueueItem]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoRoundupReport(BaseModel):
    ready_for_paper: bool
    passed_target_gates: bool
    case_count: int
    target_case_count: int
    needed_cases: int
    long_video_case_count: int
    needed_long_video_cases: int
    distinct_videos: int
    target_distinct_videos: int
    distinct_domains: int
    target_distinct_domains: int
    category_counts: dict[str, int]
    needed_by_category: dict[str, int]
    transcript_metadata_rate: float
    target_transcript_metadata_rate: float
    metadata_refresh_remaining: int
    metadata_refresh_needed_for_target: int
    target_failures: list[LongVideoSuiteGateResult]
    next_actions: list[str]
    next_curation_command: str | None = None
    next_metadata_refresh_command: str | None = None
    promotion_command_template: str | None = None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoMetadataRefreshRunItem(BaseModel):
    case_id: str
    command: str
    returncode: int
    succeeded: bool


class LongVideoMetadataRefreshRunReport(BaseModel):
    attempted: int
    succeeded: int
    failed: int
    items: list[LongVideoMetadataRefreshRunItem]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoMetadataRefreshPromotionResult(BaseModel):
    case_id: str
    promoted: bool
    quality_passed: bool
    promotion_mode: str = "full"
    source_path: Path
    target_path: Path | None = None
    failures: list[str] = Field(default_factory=list)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoArtifactAuditItem(BaseModel):
    path: Path
    curated: bool
    candidate: bool
    reasons: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    video_id: str | None = None
    query: str | None = None
    query_intent: str | None = None
    answer: str | None = None
    selected_evidence: int = 0
    visual_evidence: int = 0
    audio_evidence: int = 0
    token_reduction_percent: float | None = None
    quality_warnings: list[str] = Field(default_factory=list)


class LongVideoArtifactAuditReport(BaseModel):
    root: Path
    artifacts: int
    curated_artifacts: int
    candidate_artifacts: int
    rejected_artifacts: int
    items: list[LongVideoArtifactAuditItem]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class LongVideoSuiteReport(BaseModel):
    passed: bool
    gates: LongVideoSuiteGates
    gate_results: list[LongVideoSuiteGateResult]
    case_count: int
    long_video_case_count: int
    category_counts: dict[str, int]
    domain_counts: dict[str, int]
    video_counts: dict[str, int]
    missing_artifacts: list[Path] = Field(default_factory=list)
    metadata_failures: list[str] = Field(default_factory=list)
    health: LongVideoHealthSummary = Field(default_factory=LongVideoHealthSummary)
    expansion_plan: LongVideoExpansionPlan = Field(default_factory=LongVideoExpansionPlan)
    quality: QualityReport | None = None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def audit_long_video_artifacts(
    root: Path,
    cases: list[QualityCase],
    gates: LongVideoSuiteGates,
) -> LongVideoArtifactAuditReport:
    curated_paths = {
        case.compression_path.resolve()
        for case in cases
        if case.compression_path is not None and case.compression_path.exists()
    }
    items = [
        _audit_artifact(path=path, curated_paths=curated_paths, gates=gates)
        for path in sorted(root.rglob("compression.json"))
    ]
    return LongVideoArtifactAuditReport(
        root=root,
        artifacts=len(items),
        curated_artifacts=sum(item.curated for item in items),
        candidate_artifacts=sum(item.candidate for item in items),
        rejected_artifacts=sum(not item.curated and not item.candidate for item in items),
        items=items,
    )


def evaluate_long_video_suite(
    cases: list[QualityCase],
    gates: LongVideoSuiteGates,
    quality: QualityReport | None = None,
) -> LongVideoSuiteReport:
    category_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    missing_artifacts: list[Path] = []
    metadata_failures: list[str] = []
    artifact_payloads: list[dict] = []
    proposal_sources: list[tuple[QualityCase, Path, dict]] = []
    long_case_count = 0

    for case in cases:
        if case.query_category is None:
            metadata_failures.append(f"{case.id}: query_category is required")
        else:
            category_counts[case.query_category.value] += 1
        if not case.domain or not case.domain.strip():
            metadata_failures.append(f"{case.id}: domain is required")
        else:
            domain_counts[case.domain.strip().lower()] += 1

        artifact = case.compression_path
        if artifact is None or not artifact.exists():
            if artifact is not None:
                missing_artifacts.append(artifact)
            else:
                metadata_failures.append(
                    f"{case.id}: compression_path is required for curated suite readiness"
                )
            continue
        payload = _load_artifact(artifact)
        artifact_payloads.append(payload)
        proposal_sources.append((case, artifact, payload))
        duration_seconds, video_id = _artifact_identity(artifact)
        video_counts[video_id] += 1
        if duration_seconds >= gates.minimum_duration_seconds:
            long_case_count += 1
        else:
            metadata_failures.append(
                f"{case.id}: duration {duration_seconds:.2f}s is below "
                f"{gates.minimum_duration_seconds:.2f}s"
            )

    gate_results = [
        _at_least("cases", len(cases), gates.min_cases),
        _at_least("long_video_cases", long_case_count, gates.min_cases),
        _at_least("distinct_videos", len(video_counts), gates.min_distinct_videos),
        _at_least("distinct_domains", len(domain_counts), gates.min_distinct_domains),
    ]
    gate_results.extend(
        _at_least(
            f"category_{category.value}",
            category_counts[category.value],
            gates.min_cases_per_category,
        )
        for category in REQUIRED_QUERY_CATEGORIES
    )
    gate_results.append(
        _at_most("missing_artifacts", len(missing_artifacts), 0)
    )
    gate_results.append(
        _at_most("metadata_failures", len(metadata_failures), 0)
    )
    if quality is not None:
        gate_results.append(
            _at_least(
                "quality_pass_rate",
                quality.summary.pass_rate,
                gates.min_quality_pass_rate,
            )
        )

    health = _health_summary(artifact_payloads)
    if health.artifacts:
        gate_results.extend(
            [
                _at_least(
                    "avg_token_reduction_percent",
                    health.avg_token_reduction_percent,
                    gates.min_avg_token_reduction_percent,
                ),
                _at_most(
                    "noisy_transcript_warning_rate",
                    health.noisy_transcript_warning_rate,
                    gates.max_noisy_transcript_warning_rate,
                ),
                _at_least(
                    "transcript_metadata_rate",
                    health.transcript_metadata_rate,
                    gates.min_transcript_metadata_rate,
                ),
                _at_least("answered_rate", health.answered_rate, gates.min_answered_rate),
                _at_most(
                    "avg_selected_evidence",
                    health.avg_selected_evidence,
                    gates.max_avg_selected_evidence,
                ),
            ]
        )

    expansion_plan = _expansion_plan(
        case_count=len(cases),
        long_case_count=long_case_count,
        category_counts=category_counts,
        domain_counts=domain_counts,
        video_counts=video_counts,
        proposal_sources=proposal_sources,
        gates=gates,
    )

    return LongVideoSuiteReport(
        passed=all(result.passed for result in gate_results),
        gates=gates,
        gate_results=gate_results,
        case_count=len(cases),
        long_video_case_count=long_case_count,
        category_counts=dict(sorted(category_counts.items())),
        domain_counts=dict(sorted(domain_counts.items())),
        video_counts=dict(sorted(video_counts.items())),
        missing_artifacts=missing_artifacts,
        metadata_failures=metadata_failures,
        health=health,
        expansion_plan=expansion_plan,
        quality=quality,
    )


def curate_long_video_query_proposal(
    proposal: LongVideoQueryProposal,
    output_root: Path,
    sample_count: int | None = None,
    audio_window_seconds: float | None = None,
    visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE,
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE,
    transcript_quality: TranscriptQuality = TranscriptQuality.FAST,
    pipeline: LocalCompressionPipeline | None = None,
) -> LongVideoCurationResult:
    source_payload = _load_artifact(proposal.source_artifact)
    video_path = _artifact_source_path(source_payload, proposal.source_artifact)
    if not video_path.exists():
        raise FileNotFoundError(f"source video does not exist: {video_path}")

    run_dir = output_root / _safe_stem(proposal.video_id) / _safe_stem(proposal.query)
    run_dir.mkdir(parents=True, exist_ok=True)
    active_pipeline = pipeline or LocalCompressionPipeline(output_root=output_root)
    ingestion, compression = active_pipeline.run(
        video_path=video_path,
        query=proposal.query,
        preset=CompressionPreset.BALANCED,
        sample_count=sample_count,
        audio_window_seconds=audio_window_seconds,
        processing_mode=ProcessingMode.AUTO,
        visual_scorer=visual_scorer,
        audio_scorer=audio_scorer,
        adaptive_budget=True,
        decompose_query=True,
        task_aware_selection=True,
        visual_ocr=True,
        transcript_quality=transcript_quality,
    )

    compression_path = run_dir / "compression.json"
    compression_path.write_text(
        json.dumps(
            {
                "ingestion": ingestion.model_dump(mode="json"),
                "compression": compression.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    html_report_path = run_dir / "report.html"
    html_report_path.write_text(render_local_compression_report(ingestion, compression))

    draft = draft_quality_case(compression_path=compression_path)
    draft_case = draft.case.model_copy(
        update={
            "query_category": proposal.query_category,
            "domain": proposal.domain,
        }
    )
    draft = QualityCaseDraft(case=draft_case, notes=draft.notes)
    draft_case_path = run_dir / "quality-case.draft.jsonl"
    draft_case_path.write_text(draft.case.model_dump_json(exclude_none=True) + "\n")
    review = _curation_review(proposal=proposal, draft=draft, compression_payload=compression)
    review_json_path = run_dir / "curation-review.json"
    review_markdown_path = run_dir / "curation-review.md"
    review_json_path.write_text(review.model_dump_json(indent=2) + "\n")
    review_markdown_path.write_text(_render_curation_review_markdown(review))
    return LongVideoCurationResult(
        proposal=proposal,
        video_path=video_path,
        compression_path=compression_path,
        html_report_path=html_report_path,
        draft_case_path=draft_case_path,
        review_json_path=review_json_path,
        review_markdown_path=review_markdown_path,
        draft=draft,
    )


def review_long_video_quality_draft(draft_path: Path) -> LongVideoCurationReview:
    cases = load_quality_cases(draft_path)
    if len(cases) != 1:
        return LongVideoCurationReview(
            ready_for_dataset=False,
            warnings=[f"Expected exactly one draft case, found {len(cases)}."],
            checklist=["Keep one reviewed JSON object per draft file."],
            recommendation="Split or fix the draft before appending it to the curated dataset.",
        )
    return _review_quality_case_for_dataset(cases[0])


def build_long_video_curation_queue(
    report: LongVideoSuiteReport,
    dataset_path: Path,
    curation_output_root: Path = Path(".gist/curation"),
    visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE,
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE,
) -> LongVideoCurationQueueReport:
    items = [
        LongVideoCurationQueueItem(
            proposal_index=index,
            video_id=proposal.video_id,
            domain=proposal.domain,
            query_category=proposal.query_category,
            query=proposal.query,
            rationale=proposal.rationale,
            source_artifact=proposal.source_artifact,
            command=(
                "gist-long-video-suite "
                f"--dataset {dataset_path} "
                f"--curate-proposal-index {index} "
                f"--curation-output-root {curation_output_root} "
                f"--curation-visual-scorer {visual_scorer.value} "
                f"--curation-audio-scorer {audio_scorer.value}"
            ),
        )
        for index, proposal in enumerate(report.expansion_plan.query_proposals)
    ]
    return LongVideoCurationQueueReport(
        passed=report.passed,
        case_count=report.case_count,
        target_case_count=report.gates.min_cases,
        needed_cases=report.expansion_plan.needed_cases,
        long_video_case_count=report.long_video_case_count,
        needed_long_video_cases=report.expansion_plan.needed_long_video_cases,
        distinct_videos=len(report.video_counts),
        target_distinct_videos=report.gates.min_distinct_videos,
        distinct_domains=len(report.domain_counts),
        target_distinct_domains=report.gates.min_distinct_domains,
        category_counts=report.category_counts,
        needed_by_category=report.expansion_plan.needed_by_category,
        priority_actions=report.expansion_plan.priority_actions,
        items=items,
    )


def build_long_video_metadata_refresh_queue(
    cases: list[QualityCase],
    target_transcript_quality: TranscriptQuality = TranscriptQuality.BALANCED,
    visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE,
    output_root: Path = Path(".gist/metadata-refresh"),
) -> LongVideoMetadataRefreshQueueReport:
    items: list[LongVideoMetadataRefreshQueueItem] = []
    for case in cases:
        if case.compression_path is None or not case.compression_path.exists():
            continue
        payload = _load_artifact(case.compression_path)
        compression = payload.get("compression", payload)
        transcript_metadata = compression.get("transcript_metadata")
        current_quality = None
        if isinstance(transcript_metadata, dict) and transcript_metadata:
            current_quality = str(transcript_metadata.get("quality") or "unknown")
            continue
        if not _requires_transcript_metadata(compression):
            continue
        query = str(case.query or compression.get("query") or "").strip()
        if not query:
            continue
        source_path = _artifact_source_path(payload, case.compression_path)
        display_command_args = [
            "gist-compress",
            str(source_path),
            "--query",
            query,
            "--output-root",
            str(output_root),
            "--preset",
            str(case.preset.value),
            "--processing-mode",
            str(case.processing_mode.value),
            "--visual-scorer",
            str(visual_scorer.value),
            "--audio-scorer",
            str(AudioScoringMode.WHISPER.value),
            "--transcript-quality",
            str(target_transcript_quality.value),
            "--html-report",
            "--auto-transcript-retry",
        ]
        command_args = [sys.executable, "-m", "gist.cli", *display_command_args[1:]]
        if case.adaptive_budget:
            display_command_args.append("--adaptive-budget")
            command_args.append("--adaptive-budget")
        if case.decompose_query:
            display_command_args.append("--decompose-query")
            command_args.append("--decompose-query")
        if not case.visual_ocr:
            display_command_args.append("--no-visual-ocr")
            command_args.append("--no-visual-ocr")
        items.append(
            LongVideoMetadataRefreshQueueItem(
                case_id=case.id,
                video_id=str(compression.get("video_id") or case.id),
                query=query,
                compression_path=case.compression_path,
                source_path=source_path,
                current_transcript_quality=current_quality,
                command=shlex.join(display_command_args),
                command_args=command_args,
            )
        )
    return LongVideoMetadataRefreshQueueReport(
        cases=len(cases),
        refresh_needed=len(items),
        target_transcript_quality=target_transcript_quality,
        items=items,
    )


def build_long_video_roundup_report(
    report: LongVideoSuiteReport,
    curation_queue: LongVideoCurationQueueReport,
    metadata_refresh_queue: LongVideoMetadataRefreshQueueReport,
) -> LongVideoRoundupReport:
    current_metadata_count = int(
        round(report.health.transcript_metadata_rate * report.case_count)
    )
    target_metadata_count = ceil(
        report.gates.min_transcript_metadata_rate * report.case_count
    )
    metadata_refresh_needed_for_target = min(
        metadata_refresh_queue.refresh_needed,
        max(0, target_metadata_count - current_metadata_count),
    )
    target_failures = [result for result in report.gate_results if not result.passed]
    next_actions = list(report.expansion_plan.priority_actions)
    if metadata_refresh_needed_for_target > 0:
        next_actions.insert(
            0,
            "Refresh and promote "
            f"{metadata_refresh_needed_for_target} curated case(s) with transcript "
            "metadata to meet the metadata coverage gate.",
        )
    if not target_failures:
        next_actions = [
            "Freeze the curated long-video dataset and start paper experiments/ablations."
        ]
    promotion_template = None
    if metadata_refresh_queue.items and metadata_refresh_needed_for_target > 0:
        promotion_template = (
            "gist-long-video-suite --dataset data/eval/long-video-quality.jsonl "
            "--promote-metadata-refresh-case <case-id> "
            "--promote-metadata-refresh-compression "
            ".gist/metadata-refresh/<video-slug>/<query-slug>/compression.json "
            "--metadata-refresh-promotion-mode metadata-only "
            "--metadata-refresh-promotion-output "
            "reports/long-video-suite/metadata-refresh-promotion.json"
        )
    return LongVideoRoundupReport(
        ready_for_paper=report.passed,
        passed_target_gates=report.passed,
        case_count=report.case_count,
        target_case_count=report.gates.min_cases,
        needed_cases=report.expansion_plan.needed_cases,
        long_video_case_count=report.long_video_case_count,
        needed_long_video_cases=report.expansion_plan.needed_long_video_cases,
        distinct_videos=len(report.video_counts),
        target_distinct_videos=report.gates.min_distinct_videos,
        distinct_domains=len(report.domain_counts),
        target_distinct_domains=report.gates.min_distinct_domains,
        category_counts=report.category_counts,
        needed_by_category=report.expansion_plan.needed_by_category,
        transcript_metadata_rate=report.health.transcript_metadata_rate,
        target_transcript_metadata_rate=report.gates.min_transcript_metadata_rate,
        metadata_refresh_remaining=metadata_refresh_queue.refresh_needed,
        metadata_refresh_needed_for_target=metadata_refresh_needed_for_target,
        target_failures=target_failures,
        next_actions=next_actions,
        next_curation_command=(
            curation_queue.items[0].command if curation_queue.items else None
        ),
        next_metadata_refresh_command=(
            metadata_refresh_queue.items[0].command
            if metadata_refresh_queue.items and metadata_refresh_needed_for_target > 0
            else None
        ),
        promotion_command_template=promotion_template,
    )


def run_long_video_metadata_refresh_queue(
    queue: LongVideoMetadataRefreshQueueReport,
    limit: int | None = None,
    runner=subprocess.run,
) -> LongVideoMetadataRefreshRunReport:
    selected_items = queue.items[:limit] if limit is not None else queue.items
    run_items: list[LongVideoMetadataRefreshRunItem] = []
    for item in selected_items:
        completed = runner(item.command_args, check=False)
        returncode = int(completed.returncode)
        run_items.append(
            LongVideoMetadataRefreshRunItem(
                case_id=item.case_id,
                command=item.command,
                returncode=returncode,
                succeeded=returncode == 0,
            )
        )
    succeeded = sum(item.succeeded for item in run_items)
    return LongVideoMetadataRefreshRunReport(
        attempted=len(run_items),
        succeeded=succeeded,
        failed=len(run_items) - succeeded,
        items=run_items,
    )


def promote_long_video_metadata_refresh(
    cases: list[QualityCase],
    case_id: str,
    refreshed_compression_path: Path,
    quality_output_root: Path = Path(".gist/metadata-refresh-promotion"),
    promotion_mode: str = "full",
) -> LongVideoMetadataRefreshPromotionResult:
    if promotion_mode not in {"full", "metadata-only"}:
        raise ValueError("promotion_mode must be 'full' or 'metadata-only'")
    source_path = refreshed_compression_path
    if not source_path.exists():
        return LongVideoMetadataRefreshPromotionResult(
            case_id=case_id,
            promoted=False,
            quality_passed=False,
            promotion_mode=promotion_mode,
            source_path=source_path,
            failures=[f"refreshed compression does not exist: {source_path}"],
        )

    matches = [case for case in cases if case.id == case_id]
    if len(matches) != 1:
        return LongVideoMetadataRefreshPromotionResult(
            case_id=case_id,
            promoted=False,
            quality_passed=False,
            promotion_mode=promotion_mode,
            source_path=source_path,
            failures=[f"expected exactly one matching case, found {len(matches)}"],
        )
    case = matches[0]
    if case.compression_path is None:
        return LongVideoMetadataRefreshPromotionResult(
            case_id=case_id,
            promoted=False,
            quality_passed=False,
            promotion_mode=promotion_mode,
            source_path=source_path,
            failures=["target case has no compression_path"],
        )

    refreshed_case = case.model_copy(update={"compression_path": source_path})
    quality = run_quality_cases(
        [refreshed_case],
        output_root=quality_output_root / _safe_stem(case_id),
    )
    if promotion_mode == "metadata-only":
        target_quality = run_quality_cases(
            [case],
            output_root=quality_output_root / f"{_safe_stem(case_id)}-target",
        )
        if not target_quality.passed:
            failures = [
                failure
                for result in target_quality.results
                for failure in result.failures
            ]
            return LongVideoMetadataRefreshPromotionResult(
                case_id=case_id,
                promoted=False,
                quality_passed=False,
                promotion_mode=promotion_mode,
                source_path=source_path,
                target_path=case.compression_path,
                failures=failures or ["target artifact failed quality checks"],
            )
        refreshed_payload = _load_artifact(source_path)
        refreshed_compression = refreshed_payload.get("compression", refreshed_payload)
        transcript_metadata = refreshed_compression.get("transcript_metadata")
        if not isinstance(transcript_metadata, dict) or not transcript_metadata:
            return LongVideoMetadataRefreshPromotionResult(
                case_id=case_id,
                promoted=False,
                quality_passed=False,
                promotion_mode=promotion_mode,
                source_path=source_path,
                target_path=case.compression_path,
                failures=["refreshed artifact has no transcript_metadata to promote"],
            )
        target_payload = _load_artifact(case.compression_path)
        target_compression = target_payload.setdefault("compression", {})
        target_compression["transcript_metadata"] = transcript_metadata
        case.compression_path.write_text(json.dumps(target_payload, indent=2) + "\n")
        return LongVideoMetadataRefreshPromotionResult(
            case_id=case_id,
            promoted=True,
            quality_passed=True,
            promotion_mode=promotion_mode,
            source_path=source_path,
            target_path=case.compression_path,
        )

    if not quality.passed:
        failures = [
            failure
            for result in quality.results
            for failure in result.failures
        ]
        return LongVideoMetadataRefreshPromotionResult(
            case_id=case_id,
            promoted=False,
            quality_passed=False,
            promotion_mode=promotion_mode,
            source_path=source_path,
            target_path=case.compression_path,
            failures=failures or ["refreshed artifact failed quality checks"],
        )

    _copy_refresh_artifact_tree(source_path.parent, case.compression_path.parent)
    return LongVideoMetadataRefreshPromotionResult(
        case_id=case_id,
        promoted=True,
        quality_passed=True,
        promotion_mode=promotion_mode,
        source_path=source_path,
        target_path=case.compression_path,
    )


def _copy_refresh_artifact_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_child in source_dir.iterdir():
        target_child = target_dir / source_child.name
        if source_child.is_dir():
            if target_child.exists():
                shutil.rmtree(target_child)
            shutil.copytree(source_child, target_child)
        else:
            shutil.copy2(source_child, target_child)


def append_reviewed_long_video_quality_draft(
    draft_path: Path,
    dataset_path: Path,
) -> LongVideoDraftAppendResult:
    cases = load_quality_cases(draft_path)
    if len(cases) != 1:
        review = LongVideoCurationReview(
            ready_for_dataset=False,
            warnings=[f"Expected exactly one draft case, found {len(cases)}."],
            checklist=["Keep one reviewed JSON object per draft file."],
            recommendation="Split or fix the draft before appending it to the curated dataset.",
        )
        return LongVideoDraftAppendResult(review=review, dataset_path=dataset_path)

    case = cases[0]
    review = _review_quality_case_for_dataset(case)
    if not review.ready_for_dataset:
        return LongVideoDraftAppendResult(
            review=review,
            dataset_path=dataset_path,
            case_id=case.id,
        )

    existing_cases = load_quality_cases(dataset_path) if dataset_path.exists() else []
    if any(existing.id == case.id for existing in existing_cases):
        duplicate_review = LongVideoCurationReview(
            ready_for_dataset=False,
            warnings=[f"case id already exists in dataset: {case.id}"],
            checklist=["Choose a unique case id before appending this draft."],
            recommendation="Do not append duplicate curated cases.",
        )
        return LongVideoDraftAppendResult(
            review=duplicate_review,
            dataset_path=dataset_path,
            case_id=case.id,
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a") as handle:
        handle.write(case.model_dump_json(exclude_none=True) + "\n")
    return LongVideoDraftAppendResult(
        review=review,
        appended=True,
        dataset_path=dataset_path,
        case_id=case.id,
    )


def _audit_artifact(
    path: Path,
    curated_paths: set[Path],
    gates: LongVideoSuiteGates,
) -> LongVideoArtifactAuditItem:
    reasons: list[str] = []
    curated = path.resolve() in curated_paths
    try:
        payload = _load_artifact(path)
    except (OSError, json.JSONDecodeError) as exc:
        return LongVideoArtifactAuditItem(
            path=path,
            curated=curated,
            candidate=False,
            reasons=[f"artifact could not be read: {exc}"],
        )

    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion")
    duration_seconds: float | None = None
    if ingestion is None:
        reasons.append("missing ingestion metadata")
    else:
        try:
            duration_seconds = float(ingestion["metadata"]["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            reasons.append("missing duration metadata")

    metrics = compression.get("metrics") or {}
    selected = compression.get("selected") or []
    token_reduction = _optional_float(metrics.get("estimated_token_reduction_percent"))
    if curated:
        reasons.append("already curated")
    if duration_seconds is None or duration_seconds < gates.minimum_duration_seconds:
        actual = 0.0 if duration_seconds is None else duration_seconds
        reasons.append(
            f"duration {actual:.2f}s below {gates.minimum_duration_seconds:.2f}s"
        )
    if not str(compression.get("answer") or "").strip():
        reasons.append("missing answer")
    else:
        reasons.extend(_answer_audit_reasons(str(compression.get("answer"))))
    if not selected:
        reasons.append("no selected evidence")
    if token_reduction is None:
        reasons.append("missing token reduction metric")
    elif token_reduction < gates.min_avg_token_reduction_percent:
        reasons.append(
            f"token reduction {token_reduction:.2f}% below "
            f"{gates.min_avg_token_reduction_percent:.2f}%"
        )

    return LongVideoArtifactAuditItem(
        path=path,
        curated=curated,
        candidate=not curated and not reasons,
        reasons=reasons,
        duration_seconds=duration_seconds,
        video_id=_optional_string(compression.get("video_id")),
        query=_optional_string(compression.get("query")),
        query_intent=_optional_string(compression.get("query_intent")),
        answer=_optional_string(compression.get("answer")),
        selected_evidence=len(selected),
        visual_evidence=int(metrics.get("visual_selected") or 0),
        audio_evidence=int(metrics.get("audio_selected") or 0),
        token_reduction_percent=token_reduction,
        quality_warnings=[
            str(warning.get("code"))
            for warning in compression.get("quality_warnings", [])
            if warning.get("code")
        ],
    )


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _answer_audit_reasons(answer: str) -> list[str]:
    normalized = " ".join(answer.strip().split())
    if not normalized:
        return ["missing answer"]
    if "could not derive a reliable answer" in normalized.lower():
        return ["unreliable generated answer"]
    if _looks_like_noisy_ocr_answer(normalized):
        return ["answer appears OCR-noisy"]
    return []


def _looks_like_noisy_ocr_answer(answer: str) -> bool:
    marker = "on-screen text near"
    if marker not in answer.lower():
        return False
    _, _, ocr_text = answer.partition(":")
    content_terms = [
        term.lower()
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", ocr_text)
        if term.lower() not in {"near", "seconds", "screen", "text"}
    ]
    if len(content_terms) < 2:
        return True
    short_terms = sum(len(term) <= 3 for term in content_terms)
    return short_terms / len(content_terms) > 0.6


def render_long_video_suite_markdown(report: LongVideoSuiteReport) -> str:
    gate_rows = "\n".join(
        f"| {result.name} | {'pass' if result.passed else 'fail'} | "
        f"{result.actual:.2f} | {result.required:.2f} |"
        for result in report.gate_results
    )
    category_rows = "\n".join(
        f"| {category.value} | {report.category_counts.get(category.value, 0)} | "
        f"{report.gates.min_cases_per_category} |"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    quality_detail = (
        f"\n## Quality Detail\n\n{render_quality_markdown(report.quality)}"
        if report.quality is not None
        else ""
    )
    failures = [
        *(f"Missing artifact: `{path}`" for path in report.missing_artifacts),
        *report.metadata_failures,
    ]
    failure_lines = "\n".join(f"- {failure}" for failure in failures) or "- None"
    warning_rows = "\n".join(
        f"| {code} | {count} |"
        for code, count in report.health.quality_warning_counts.items()
    ) or "| none | 0 |"
    transcript_rows = "\n".join(
        f"| {quality} | {count} |"
        for quality, count in report.health.transcript_quality_counts.items()
    ) or "| none | 0 |"
    missing_category_rows = "\n".join(
        f"| {category} | {needed} |"
        for category, needed in report.expansion_plan.needed_by_category.items()
    ) or "| none | 0 |"
    priority_lines = "\n".join(
        f"- {action}" for action in report.expansion_plan.priority_actions
    ) or "- No expansion actions required by the configured gates."
    proposal_rows = "\n".join(
        f"| {proposal.video_id} | {proposal.query_category.value} | "
        f"{proposal.query} | {proposal.rationale} |"
        for proposal in report.expansion_plan.query_proposals
    ) or "| none | none | No proposal needed. | - |"
    return f"""# Gist Long-Video Suite Readiness

- Passed: {"yes" if report.passed else "no"}
- Cases: {report.case_count}
- Valid long-video cases: {report.long_video_case_count}
- Distinct videos: {len(report.video_counts)}
- Distinct domains: {len(report.domain_counts)}

## Gates

| Gate | Status | Actual | Required |
|---|---:|---:|---:|
{gate_rows}

## Query Coverage

| Category | Cases | Required |
|---|---:|---:|
{category_rows}

## Run Health

- Artifacts: {report.health.artifacts}
- Avg token reduction: {report.health.avg_token_reduction_percent:.2f}%
- Noisy transcript warning rate: {report.health.noisy_transcript_warning_rate:.2%}
- Transcript metadata rate: {report.health.transcript_metadata_rate:.2%}
- Answered rate: {report.health.answered_rate:.2%}
- Avg selected evidence: {report.health.avg_selected_evidence:.2f}

### Quality Warnings

| Warning | Count |
|---|---:|
{warning_rows}

### Transcript Quality

| Quality | Count |
|---|---:|
{transcript_rows}

## Expansion Plan

- Additional cases needed: {report.expansion_plan.needed_cases}
- Additional long-video cases needed: {report.expansion_plan.needed_long_video_cases}
- Additional distinct videos needed: {report.expansion_plan.needed_distinct_videos}
- Additional distinct domains needed: {report.expansion_plan.needed_distinct_domains}

### Missing Query Categories

| Category | Additional Cases Needed |
|---|---:|
{missing_category_rows}

### Priority Actions

{priority_lines}

### Query Proposals

| Video | Category | Proposed Query | Rationale |
|---|---|---|---|
{proposal_rows}

## Dataset Problems

{failure_lines}
{quality_detail}
"""


def render_long_video_suite_html(report: LongVideoSuiteReport) -> str:
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(result.name)}</td>"
        f"<td>{'pass' if result.passed else 'fail'}</td>"
        f"<td>{result.actual:.2f}</td>"
        f"<td>{result.required:.2f}</td>"
        "</tr>"
        for result in report.gate_results
    )
    category_rows = "\n".join(
        "<tr>"
        f"<td>{escape(category.value)}</td>"
        f"<td>{report.category_counts.get(category.value, 0)}</td>"
        f"<td>{report.gates.min_cases_per_category}</td>"
        "</tr>"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    problems = [
        *(f"Missing artifact: {path}" for path in report.missing_artifacts),
        *report.metadata_failures,
    ]
    problem_items = (
        "".join(f"<li>{escape(problem)}</li>" for problem in problems)
        or "<li>None</li>"
    )
    warning_rows = "".join(
        f"<tr><td>{escape(code)}</td><td>{count}</td></tr>"
        for code, count in report.health.quality_warning_counts.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    transcript_rows = "".join(
        f"<tr><td>{escape(quality)}</td><td>{count}</td></tr>"
        for quality, count in report.health.transcript_quality_counts.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    missing_category_rows = "".join(
        f"<tr><td>{escape(category)}</td><td>{needed}</td></tr>"
        for category, needed in report.expansion_plan.needed_by_category.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    priority_items = (
        "".join(
            f"<li>{escape(action)}</li>"
            for action in report.expansion_plan.priority_actions
        )
        or "<li>No expansion actions required by the configured gates.</li>"
    )
    proposal_rows = "".join(
        "<tr>"
        f"<td>{escape(proposal.video_id)}</td>"
        f"<td>{escape(proposal.query_category.value)}</td>"
        f"<td>{escape(proposal.query)}</td>"
        f"<td>{escape(proposal.rationale)}</td>"
        "</tr>"
        for proposal in report.expansion_plan.query_proposals
    ) or "<tr><td>none</td><td>none</td><td>No proposal needed.</td><td>-</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Long-Video Suite Readiness</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: min(920px, 100%); margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf5f3; }}
  </style>
</head>
<body>
  <h1>Gist Long-Video Suite Readiness</h1>
  <p><strong>Status:</strong> {'pass' if report.passed else 'fail'}</p>
  <h2>Gates</h2>
  <table>
    <tr><th>Gate</th><th>Status</th><th>Actual</th><th>Required</th></tr>
    {gate_rows}
  </table>
  <h2>Query Coverage</h2>
  <table>
    <tr><th>Category</th><th>Cases</th><th>Required</th></tr>
    {category_rows}
  </table>
  <h2>Run Health</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Artifacts</td><td>{report.health.artifacts}</td></tr>
    <tr><td>Avg token reduction</td><td>{report.health.avg_token_reduction_percent:.2f}%</td></tr>
    <tr><td>Noisy transcript warning rate</td><td>
      {report.health.noisy_transcript_warning_rate:.2%}
    </td></tr>
    <tr><td>Transcript metadata rate</td><td>{report.health.transcript_metadata_rate:.2%}</td></tr>
    <tr><td>Answered rate</td><td>{report.health.answered_rate:.2%}</td></tr>
    <tr><td>Avg selected evidence</td><td>{report.health.avg_selected_evidence:.2f}</td></tr>
  </table>
  <h2>Quality Warnings</h2>
  <table>
    <tr><th>Warning</th><th>Count</th></tr>
    {warning_rows}
  </table>
  <h2>Transcript Quality</h2>
  <table>
    <tr><th>Quality</th><th>Count</th></tr>
    {transcript_rows}
  </table>
  <h2>Expansion Plan</h2>
  <table>
    <tr><th>Metric</th><th>Needed</th></tr>
    <tr><td>Additional cases</td><td>{report.expansion_plan.needed_cases}</td></tr>
    <tr><td>Additional long-video cases</td><td>{report.expansion_plan.needed_long_video_cases}</td></tr>
    <tr><td>Additional distinct videos</td><td>{report.expansion_plan.needed_distinct_videos}</td></tr>
    <tr><td>Additional distinct domains</td><td>{report.expansion_plan.needed_distinct_domains}</td></tr>
  </table>
  <h3>Missing Query Categories</h3>
  <table>
    <tr><th>Category</th><th>Additional Cases Needed</th></tr>
    {missing_category_rows}
  </table>
  <h3>Priority Actions</h3>
  <ul>{priority_items}</ul>
  <h3>Query Proposals</h3>
  <table>
    <tr><th>Video</th><th>Category</th><th>Proposed Query</th><th>Rationale</th></tr>
    {proposal_rows}
  </table>
  <h2>Dataset Problems</h2>
  <ul>{problem_items}</ul>
</body>
</html>
"""


def render_long_video_curation_queue_markdown(
    queue: LongVideoCurationQueueReport,
) -> str:
    category_rows = "\n".join(
        f"| {category.value} | {queue.category_counts.get(category.value, 0)} | "
        f"{queue.needed_by_category.get(category.value, 0)} |"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    priority_lines = "\n".join(
        f"- {action}" for action in queue.priority_actions
    ) or "- No curation actions required by the configured gates."
    queue_rows = "\n".join(
        f"| {item.proposal_index} | {item.video_id} | {item.query_category.value} | "
        f"{item.query} | `{item.command}` |"
        for item in queue.items
    ) or "| - | - | - | No proposal needed. | - |"
    next_step = (
        f"Run proposal `{queue.items[0].proposal_index}` first, then review and append its draft."
        if queue.items
        else "No queued curation proposal is needed for the configured gates."
    )
    return f"""# Gist Long-Video Curation Queue

- Passed target gates: {"yes" if queue.passed else "no"}
- Cases: {queue.case_count}/{queue.target_case_count}
- Long-video cases: {queue.long_video_case_count}/{queue.target_case_count}
- Distinct videos: {queue.distinct_videos}/{queue.target_distinct_videos}
- Distinct domains: {queue.distinct_domains}/{queue.target_distinct_domains}
- Additional cases needed: {queue.needed_cases}
- Additional long-video cases needed: {queue.needed_long_video_cases}

## Category Coverage

| Category | Current Cases | Additional Needed |
|---|---:|---:|
{category_rows}

## Priority Actions

{priority_lines}

## Queue

| Index | Video | Category | Query | Command |
|---:|---|---|---|---|
{queue_rows}

## Next Step

{next_step}
"""


def render_long_video_metadata_refresh_queue_markdown(
    queue: LongVideoMetadataRefreshQueueReport,
) -> str:
    rows = "\n".join(
        f"| {item.case_id} | {item.video_id} | {item.query} | `{item.command}` |"
        for item in queue.items
    ) or "| - | - | No refresh needed. | - |"
    next_step = (
        f"Regenerate `{queue.items[0].case_id}` first, then rerun the long-video suite."
        if queue.items
        else "No transcript metadata refresh is needed for the current curated cases."
    )
    return f"""# Gist Long-Video Transcript Metadata Refresh Queue

- Cases: {queue.cases}
- Refresh needed: {queue.refresh_needed}
- Target transcript quality: {queue.target_transcript_quality.value}

## Queue

| Case | Video | Query | Command |
|---|---|---|---|
{rows}

## Next Step

{next_step}
"""


def render_long_video_roundup_markdown(report: LongVideoRoundupReport) -> str:
    category_rows = "\n".join(
        f"| {category.value} | {report.category_counts.get(category.value, 0)} | "
        f"{report.needed_by_category.get(category.value, 0)} |"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    failure_rows = "\n".join(
        f"| {failure.name} | {failure.actual:.2f} | {failure.required:.2f} | "
        f"{failure.message} |"
        for failure in report.target_failures
    ) or "| - | - | - | No target gate failures. |"
    action_lines = "\n".join(
        f"{index}. {action}" for index, action in enumerate(report.next_actions, start=1)
    ) or "1. No next action required."
    curation_command = (
        f"`{report.next_curation_command}`"
        if report.next_curation_command
        else "No curation command required."
    )
    metadata_command = (
        f"`{report.next_metadata_refresh_command}`"
        if report.next_metadata_refresh_command
        else "No metadata refresh command required."
    )
    promotion_command = (
        f"`{report.promotion_command_template}`"
        if report.promotion_command_template
        else "No metadata promotion command required."
    )
    return f"""# Gist Long-Video Roundup

- Ready for paper freeze: {"yes" if report.ready_for_paper else "no"}
- Passed target gates: {"yes" if report.passed_target_gates else "no"}
- Cases: {report.case_count}/{report.target_case_count}
- Long-video cases: {report.long_video_case_count}/{report.target_case_count}
- Distinct videos: {report.distinct_videos}/{report.target_distinct_videos}
- Distinct domains: {report.distinct_domains}/{report.target_distinct_domains}
- Transcript metadata rate: {report.transcript_metadata_rate:.2f}/{report.target_transcript_metadata_rate:.2f}
- Metadata refreshes queued: {report.metadata_refresh_remaining}
- Metadata refreshes needed for target: {report.metadata_refresh_needed_for_target}

## Category Coverage

| Category | Current Cases | Additional Needed |
|---|---:|---:|
{category_rows}

## Target Failures

| Gate | Actual | Required | Message |
|---|---:|---:|---|
{failure_rows}

## Next Actions

{action_lines}

## Commands

- Next curation: {curation_command}
- Next metadata refresh: {metadata_command}
- Promotion template: {promotion_command}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check coverage and optionally run quality for the curated long-video suite."
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument("--queue-markdown-output", type=Path)
    parser.add_argument("--roundup-output", type=Path)
    parser.add_argument("--roundup-markdown-output", type=Path)
    parser.add_argument(
        "--print-next-actions",
        action="store_true",
        help=(
            "Print the long-video roundup markdown to stdout, including remaining "
            "gates and exact next curation/metadata commands."
        ),
    )
    parser.add_argument("--metadata-refresh-output", type=Path)
    parser.add_argument("--metadata-refresh-markdown-output", type=Path)
    parser.add_argument("--metadata-refresh-run-output", type=Path)
    parser.add_argument(
        "--metadata-refresh-output-root",
        type=Path,
        default=Path(".gist/metadata-refresh"),
        help="Output root for refreshed artifacts; defaults to a non-destructive review area.",
    )
    parser.add_argument(
        "--run-metadata-refresh",
        action="store_true",
        help="Execute queued transcript metadata refresh commands.",
    )
    parser.add_argument(
        "--metadata-refresh-limit",
        type=int,
        help="Maximum number of queued metadata refresh commands to execute.",
    )
    parser.add_argument(
        "--metadata-refresh-quality",
        choices=list(TranscriptQuality),
        default=TranscriptQuality.BALANCED,
    )
    parser.add_argument(
        "--metadata-refresh-visual-scorer",
        choices=list(VisualScoringMode),
        default=VisualScoringMode.BASELINE,
        help="Visual scorer used for metadata refresh reruns; baseline avoids CLIP downloads.",
    )
    parser.add_argument("--promote-metadata-refresh-case")
    parser.add_argument("--promote-metadata-refresh-compression", type=Path)
    parser.add_argument(
        "--metadata-refresh-promotion-mode",
        choices=("full", "metadata-only"),
        default="full",
        help=(
            "Use full to replace the curated artifact only if the refreshed artifact "
            "passes quality; use metadata-only to stamp transcript metadata onto the "
            "existing curated artifact when full replacement is weaker."
        ),
    )
    parser.add_argument("--metadata-refresh-promotion-output", type=Path)
    parser.add_argument(
        "--metadata-refresh-promotion-quality-root",
        type=Path,
        default=Path(".gist/metadata-refresh-promotion"),
    )
    parser.add_argument(
        "--review-draft",
        type=Path,
        help="Validate one quality-case draft before appending it to the curated dataset.",
    )
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--review-markdown-output", type=Path)
    parser.add_argument(
        "--append-draft-to",
        type=Path,
        help="Append a reviewed --review-draft case to this dataset only if it passes readiness checks.",
    )
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/long-video-suite"))
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument(
        "--curate-proposal-index",
        type=int,
        help=(
            "Run a query proposal by zero-based index from the current expansion plan "
            "and write a review bundle with compression.json, report.html, and draft JSONL."
        ),
    )
    parser.add_argument(
        "--curation-output-root",
        type=Path,
        default=Path(".gist/curation"),
    )
    parser.add_argument("--curation-sample-count", type=int)
    parser.add_argument("--curation-audio-window-seconds", type=float)
    parser.add_argument(
        "--curation-visual-scorer",
        choices=list(VisualScoringMode),
        default=VisualScoringMode.CLIP_SCENE,
    )
    parser.add_argument(
        "--curation-audio-scorer",
        choices=list(AudioScoringMode),
        default=AudioScoringMode.BASELINE,
    )
    parser.add_argument(
        "--curation-transcript-quality",
        choices=list(TranscriptQuality),
        default=TranscriptQuality.FAST,
    )
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--min-distinct-videos", type=int, default=5)
    parser.add_argument("--min-distinct-domains", type=int, default=3)
    parser.add_argument("--min-cases-per-category", type=int, default=3)
    parser.add_argument("--minimum-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-avg-token-reduction-percent", type=float, default=90.0)
    parser.add_argument("--max-noisy-transcript-warning-rate", type=float, default=0.25)
    parser.add_argument("--min-transcript-metadata-rate", type=float, default=0.8)
    parser.add_argument("--min-answered-rate", type=float, default=0.9)
    parser.add_argument("--max-avg-selected-evidence", type=float, default=8.0)
    args = parser.parse_args(argv)

    if args.metadata_refresh_run_output is not None and not args.run_metadata_refresh:
        raise SystemExit("--metadata-refresh-run-output requires --run-metadata-refresh")
    if (args.promote_metadata_refresh_case is None) != (
        args.promote_metadata_refresh_compression is None
    ):
        raise SystemExit(
            "--promote-metadata-refresh-case and "
            "--promote-metadata-refresh-compression must be used together"
        )

    if args.review_draft is not None:
        append_result = None
        if args.append_draft_to is not None:
            append_result = append_reviewed_long_video_quality_draft(
                draft_path=args.review_draft,
                dataset_path=args.append_draft_to,
            )
            review = append_result.review
        else:
            review = review_long_video_quality_draft(args.review_draft)
        if args.review_output is not None:
            review.write_json(args.review_output)
        if args.review_markdown_output is not None:
            args.review_markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.review_markdown_output.write_text(_render_curation_review_markdown(review))
        print(f"ready_for_dataset={'yes' if review.ready_for_dataset else 'no'}")
        if append_result is not None:
            print(f"appended={'yes' if append_result.appended else 'no'}")
            print(f"dataset={append_result.dataset_path}")
            if append_result.case_id is not None:
                print(f"case_id={append_result.case_id}")
        print(f"warnings={len(review.warnings)}")
        for warning in review.warnings:
            print(f"  - {warning}")
        if append_result is not None:
            return 0 if append_result.appended else 1
        return 0 if review.ready_for_dataset else 1

    if args.dataset is None:
        raise SystemExit("--dataset is required unless --review-draft is used")

    cases = load_quality_cases(args.dataset)
    promotion_result = None
    if args.promote_metadata_refresh_case is not None:
        promotion_result = promote_long_video_metadata_refresh(
            cases=cases,
            case_id=args.promote_metadata_refresh_case,
            refreshed_compression_path=args.promote_metadata_refresh_compression,
            quality_output_root=args.metadata_refresh_promotion_quality_root,
            promotion_mode=args.metadata_refresh_promotion_mode,
        )
        if args.metadata_refresh_promotion_output is not None:
            promotion_result.write_json(args.metadata_refresh_promotion_output)
        print(f"metadata_refresh_promoted={'yes' if promotion_result.promoted else 'no'}")
        print(f"metadata_refresh_promotion_mode={promotion_result.promotion_mode}")
        print(
            "metadata_refresh_quality_passed="
            f"{'yes' if promotion_result.quality_passed else 'no'}"
        )
        if promotion_result.target_path is not None:
            print(f"metadata_refresh_target={promotion_result.target_path}")
        for failure in promotion_result.failures:
            print(f"  - {failure}")
    quality = (
        run_quality_cases(cases, output_root=args.output_root)
        if args.run_quality
        else None
    )
    report = evaluate_long_video_suite(
        cases=cases,
        gates=LongVideoSuiteGates(
            min_cases=args.min_cases,
            min_distinct_videos=args.min_distinct_videos,
            min_distinct_domains=args.min_distinct_domains,
            min_cases_per_category=args.min_cases_per_category,
            minimum_duration_seconds=args.minimum_duration_seconds,
            min_quality_pass_rate=args.min_quality_pass_rate,
            min_avg_token_reduction_percent=args.min_avg_token_reduction_percent,
            max_noisy_transcript_warning_rate=args.max_noisy_transcript_warning_rate,
            min_transcript_metadata_rate=args.min_transcript_metadata_rate,
            min_answered_rate=args.min_answered_rate,
            max_avg_selected_evidence=args.max_avg_selected_evidence,
        ),
        quality=quality,
    )
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_long_video_suite_markdown(report))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_long_video_suite_html(report))
    queue = None
    queue_requested = (
        args.queue_output is not None
        or args.queue_markdown_output is not None
        or args.roundup_output is not None
        or args.roundup_markdown_output is not None
        or args.print_next_actions
    )
    if queue_requested:
        queue = build_long_video_curation_queue(
            report=report,
            dataset_path=args.dataset,
            curation_output_root=args.curation_output_root,
            visual_scorer=VisualScoringMode(args.curation_visual_scorer),
            audio_scorer=AudioScoringMode(args.curation_audio_scorer),
        )
        if args.queue_output is not None:
            queue.write_json(args.queue_output)
        if args.queue_markdown_output is not None:
            args.queue_markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.queue_markdown_output.write_text(
                render_long_video_curation_queue_markdown(queue)
            )
        if args.queue_output is not None or args.queue_markdown_output is not None:
            print(f"queue_items={len(queue.items)}")
    metadata_refresh = None
    metadata_refresh_requested = (
        args.metadata_refresh_output is not None
        or args.metadata_refresh_markdown_output is not None
        or args.run_metadata_refresh
        or args.roundup_output is not None
        or args.roundup_markdown_output is not None
        or args.print_next_actions
    )
    metadata_refresh_run = None
    if metadata_refresh_requested:
        metadata_refresh = build_long_video_metadata_refresh_queue(
            cases=cases,
            target_transcript_quality=TranscriptQuality(args.metadata_refresh_quality),
            visual_scorer=VisualScoringMode(args.metadata_refresh_visual_scorer),
            output_root=args.metadata_refresh_output_root,
        )
        if args.metadata_refresh_output is not None:
            metadata_refresh.write_json(args.metadata_refresh_output)
        if args.metadata_refresh_markdown_output is not None:
            args.metadata_refresh_markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_refresh_markdown_output.write_text(
                render_long_video_metadata_refresh_queue_markdown(metadata_refresh)
            )
        if (
            args.metadata_refresh_output is not None
            or args.metadata_refresh_markdown_output is not None
            or args.run_metadata_refresh
        ):
            print(f"metadata_refresh_items={len(metadata_refresh.items)}")
        if args.run_metadata_refresh:
            if args.metadata_refresh_limit is not None and args.metadata_refresh_limit < 1:
                raise SystemExit("--metadata-refresh-limit must be greater than 0")
            metadata_refresh_run = run_long_video_metadata_refresh_queue(
                queue=metadata_refresh,
                limit=args.metadata_refresh_limit,
            )
            if args.metadata_refresh_run_output is not None:
                metadata_refresh_run.write_json(args.metadata_refresh_run_output)
            print(f"metadata_refresh_attempted={metadata_refresh_run.attempted}")
            print(f"metadata_refresh_succeeded={metadata_refresh_run.succeeded}")
            print(f"metadata_refresh_failed={metadata_refresh_run.failed}")
    if (
        args.roundup_output is not None
        or args.roundup_markdown_output is not None
        or args.print_next_actions
    ):
        if queue is None:
            raise RuntimeError("roundup requires a curation queue")
        if metadata_refresh is None:
            raise RuntimeError("roundup requires a metadata refresh queue")
        roundup = build_long_video_roundup_report(
            report=report,
            curation_queue=queue,
            metadata_refresh_queue=metadata_refresh,
        )
        if args.roundup_output is not None:
            roundup.write_json(args.roundup_output)
        if args.roundup_markdown_output is not None:
            args.roundup_markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.roundup_markdown_output.write_text(
                render_long_video_roundup_markdown(roundup)
            )
        if args.print_next_actions:
            print(render_long_video_roundup_markdown(roundup))
        print(f"roundup_actions={len(roundup.next_actions)}")
    curation_requested = args.curate_proposal_index is not None
    if curation_requested:
        proposals = report.expansion_plan.query_proposals
        if args.curate_proposal_index < 0 or args.curate_proposal_index >= len(proposals):
            raise SystemExit(
                f"--curate-proposal-index must be between 0 and {len(proposals) - 1}"
            )
        curation = curate_long_video_query_proposal(
            proposal=proposals[args.curate_proposal_index],
            output_root=args.curation_output_root,
            sample_count=args.curation_sample_count,
            audio_window_seconds=args.curation_audio_window_seconds,
            visual_scorer=VisualScoringMode(args.curation_visual_scorer),
            audio_scorer=AudioScoringMode(args.curation_audio_scorer),
            transcript_quality=TranscriptQuality(args.curation_transcript_quality),
        )
        print(f"curation_compression={curation.compression_path}")
        print(f"curation_report={curation.html_report_path}")
        print(f"curation_draft={curation.draft_case_path}")
        print(f"curation_review={curation.review_markdown_path}")
    if args.audit_root is not None or args.audit_output is not None:
        audit = audit_long_video_artifacts(
            root=args.audit_root or Path(".gist/runs"),
            cases=cases,
            gates=report.gates,
        )
        if args.audit_output is not None:
            audit.write_json(args.audit_output)
        print(f"audit_artifacts={audit.artifacts}")
        print(f"audit_candidates={audit.candidate_artifacts}")
        print(f"audit_rejected={audit.rejected_artifacts}")

    print(f"passed={'yes' if report.passed else 'no'}")
    print(f"cases={len(cases)}")
    print(f"distinct_videos={len(report.video_counts)}")
    print(f"distinct_domains={len(report.domain_counts)}")
    for category in REQUIRED_QUERY_CATEGORIES:
        print(f"{category.value}={report.category_counts.get(category.value, 0)}")
    for result in report.gate_results:
        if not result.passed:
            print(
                f"  - {result.name}: actual={result.actual:.2f}, "
                f"required={result.required:.2f}"
            )
    if curation_requested:
        return 0
    if metadata_refresh_run is not None:
        return 0 if metadata_refresh_run.failed == 0 else 1
    if promotion_result is not None:
        return 0 if promotion_result.promoted else 1
    return 0 if report.passed else 1


def _artifact_identity(path: Path) -> tuple[float, str]:
    payload = _load_artifact(path)
    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion")
    if ingestion is None:
        raise ValueError(f"{path}: ingestion metadata is required")
    return (
        float(ingestion["metadata"]["duration_seconds"]),
        str(compression["video_id"]),
    )


def _load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())


def _health_summary(payloads: list[dict]) -> LongVideoHealthSummary:
    if not payloads:
        return LongVideoHealthSummary()

    warning_counts: Counter[str] = Counter()
    transcript_quality_counts: Counter[str] = Counter()
    token_reductions: list[float] = []
    selected_counts: list[int] = []
    noisy_count = 0
    metadata_count = 0
    metadata_eligible_count = 0
    answered_count = 0

    for payload in payloads:
        compression = payload.get("compression", payload)
        metrics = compression.get("metrics", {})
        token_reductions.append(float(metrics.get("estimated_token_reduction_percent", 0.0)))
        selected = compression.get("selected") or []
        selected_counts.append(int(metrics.get("selected_candidates", len(selected))))
        answer = str(compression.get("answer") or "").strip()
        if len(answer.split()) >= 3:
            answered_count += 1

        warnings = compression.get("quality_warnings") or []
        warning_codes = [
            str(warning.get("code"))
            for warning in warnings
            if isinstance(warning, dict) and warning.get("code")
        ]
        warning_counts.update(warning_codes)
        if "noisy_transcript_evidence" in warning_codes:
            noisy_count += 1

        transcript_metadata = compression.get("transcript_metadata")
        requires_transcript_metadata = _requires_transcript_metadata(compression)
        if requires_transcript_metadata:
            metadata_eligible_count += 1
        if isinstance(transcript_metadata, dict) and transcript_metadata:
            metadata_count += 1
            quality = str(transcript_metadata.get("quality") or "unknown")
            transcript_quality_counts[quality] += 1

    total = len(payloads)
    return LongVideoHealthSummary(
        artifacts=total,
        avg_token_reduction_percent=_average(token_reductions),
        noisy_transcript_warning_rate=noisy_count / total,
        transcript_metadata_rate=(
            1.0 if metadata_eligible_count == 0 else metadata_count / metadata_eligible_count
        ),
        answered_rate=answered_count / total,
        avg_selected_evidence=_average(selected_counts),
        quality_warning_counts=dict(sorted(warning_counts.items())),
        transcript_quality_counts=dict(sorted(transcript_quality_counts.items())),
    )


def _requires_transcript_metadata(compression: dict) -> bool:
    if isinstance(compression.get("transcript_metadata"), dict):
        return True
    if str(compression.get("audio_scorer_used") or "").lower() == AudioScoringMode.WHISPER.value:
        return True
    selected = compression.get("selected") or []
    return any(
        isinstance(candidate, dict)
        and str(candidate.get("modality") or "").lower() == "audio"
        for candidate in selected
    )


def _expansion_plan(
    *,
    case_count: int,
    long_case_count: int,
    category_counts: Counter[str],
    domain_counts: Counter[str],
    video_counts: Counter[str],
    proposal_sources: list[tuple[QualityCase, Path, dict]],
    gates: LongVideoSuiteGates,
) -> LongVideoExpansionPlan:
    needed_by_category = {
        category.value: gates.min_cases_per_category - category_counts[category.value]
        for category in REQUIRED_QUERY_CATEGORIES
        if category_counts[category.value] < gates.min_cases_per_category
    }
    plan = LongVideoExpansionPlan(
        needed_cases=max(0, gates.min_cases - case_count),
        needed_long_video_cases=max(0, gates.min_cases - long_case_count),
        needed_distinct_videos=max(0, gates.min_distinct_videos - len(video_counts)),
        needed_distinct_domains=max(0, gates.min_distinct_domains - len(domain_counts)),
        needed_by_category=needed_by_category,
    )
    plan.priority_actions = _priority_actions(plan)
    plan.query_proposals = _query_proposals(
        needed_by_category=needed_by_category,
        proposal_sources=proposal_sources,
        category_counts=category_counts,
        needed_cases=plan.needed_cases,
    )
    return plan


def _priority_actions(plan: LongVideoExpansionPlan) -> list[str]:
    actions: list[str] = []
    if plan.needed_cases:
        actions.append(f"Curate {plan.needed_cases} more verified long-video question cases.")
    if plan.needed_long_video_cases:
        actions.append(
            f"Ensure {plan.needed_long_video_cases} added cases come from videos at least 60 minutes long."
        )
    if plan.needed_distinct_videos:
        actions.append(
            f"Add cases from {plan.needed_distinct_videos} new long-video source(s)."
        )
    if plan.needed_distinct_domains:
        actions.append(
            f"Add at least {plan.needed_distinct_domains} new domain label(s) beyond the current set."
        )
    for category, needed in plan.needed_by_category.items():
        actions.append(f"Add {needed} verified `{category}` case(s).")
    return actions


def _query_proposals(
    *,
    needed_by_category: dict[str, int],
    proposal_sources: list[tuple[QualityCase, Path, dict]],
    category_counts: Counter[str],
    needed_cases: int,
) -> list[LongVideoQueryProposal]:
    if not proposal_sources:
        return []

    categories = [
        QueryIntent(category)
        for category in needed_by_category
    ]
    if not categories and needed_cases:
        categories = sorted(
            REQUIRED_QUERY_CATEGORIES,
            key=lambda category: category_counts[category.value],
        )
    if not categories:
        return []

    proposals: list[LongVideoQueryProposal] = []
    max_proposals = max(1, min(12, needed_cases or sum(needed_by_category.values()) or len(categories)))
    used_pairs: set[tuple[str, QueryIntent]] = set()
    for category in categories:
        for case, path, payload in _ranked_proposal_sources(category, proposal_sources):
            video_id = _proposal_video_label(path, payload)
            used_pairs.add((video_id, category))
            proposals.append(
                LongVideoQueryProposal(
                    video_id=video_id,
                    domain=(case.domain or "unknown").strip() or "unknown",
                    query_category=category,
                    query=_proposal_query(category, video_id, payload),
                    rationale=_proposal_rationale(category, category_counts, needed_by_category),
                    source_artifact=path,
                )
            )
            break
        if len(proposals) >= max_proposals:
            return proposals
    for category in categories:
        seen_videos: set[str] = set()
        for case, path, payload in _ranked_proposal_sources(category, proposal_sources):
            video_id = _proposal_video_label(path, payload)
            if (video_id, category) in used_pairs:
                continue
            if video_id in seen_videos:
                continue
            seen_videos.add(video_id)
            if len(proposals) >= max_proposals:
                return proposals
            proposals.append(
                LongVideoQueryProposal(
                    video_id=video_id,
                    domain=(case.domain or "unknown").strip() or "unknown",
                    query_category=category,
                    query=_proposal_query(category, video_id, payload),
                    rationale=_proposal_rationale(category, category_counts, needed_by_category),
                    source_artifact=path,
                )
            )
    return proposals


def _ranked_proposal_sources(
    category: QueryIntent,
    proposal_sources: list[tuple[QualityCase, Path, dict]],
) -> list[tuple[QualityCase, Path, dict]]:
    return sorted(
        proposal_sources,
        key=lambda source: (
            -_proposal_source_score(category, source[0], source[2]),
            _proposal_video_label(source[1], source[2]),
            source[0].id,
        ),
    )


def _proposal_source_score(category: QueryIntent, case: QualityCase, payload: dict) -> int:
    compression = payload.get("compression", payload)
    selected = compression.get("selected") or []
    modalities = {
        str(candidate.get("modality") or "").lower()
        for candidate in selected
        if isinstance(candidate, dict)
    }
    score = 0
    if case.query_category == category:
        score += 10
    if category == QueryIntent.MIXED_AV:
        if "audio" in modalities:
            score += 6
        if "visual" in modalities:
            score += 3
        if str(compression.get("audio_scorer_used") or "").lower() == AudioScoringMode.WHISPER.value:
            score += 2
    elif category == QueryIntent.VISUAL_OBJECT_ACTION and "visual" in modalities:
        score += 5
    elif category == QueryIntent.SPEECH_SEMANTIC and "audio" in modalities:
        score += 5
    return score


def _proposal_query(category: QueryIntent, video_label: str, payload: dict) -> str:
    video_id = video_label.replace("-", " ")
    if category == QueryIntent.SPEECH_SEMANTIC:
        return f"What key claim does the speaker make in {video_id}?"
    if category == QueryIntent.VISUAL_OBJECT_ACTION:
        return f"What important object, slide, or on-screen action appears in {video_id}?"
    if category == QueryIntent.TEMPORAL_BEFORE_AFTER:
        return f"What happens immediately after an important transition in {video_id}?"
    if category == QueryIntent.GLOBAL_SUMMARY:
        return f"What are the main topics covered throughout {video_id}?"
    if category == QueryIntent.MIXED_AV:
        phrase = _proposal_audio_phrase(payload)
        if phrase:
            return f"What does the speaker say about {phrase} while the related visual moment is shown?"
        return f"What does the speaker say while the relevant visual moment is shown in {video_id}?"
    return f"What evidence in {video_id} best answers the user question?"


def _proposal_audio_phrase(payload: dict) -> str:
    compression = payload.get("compression", payload)
    for candidate in compression.get("selected") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("modality") or "").lower() != "audio":
            continue
        text = re.sub(r"[^A-Za-z0-9 ]+", " ", str(candidate.get("text") or ""))
        words = [
            word.lower()
            for word in text.split()
            if len(word) > 3 and word.lower() not in _PROPOSAL_STOPWORDS
        ]
        if len(words) >= 3:
            return " ".join(words[:6])
    return ""


def _proposal_video_label(path: Path, payload: dict) -> str:
    parts = path.parts
    if ".gist" in parts:
        gist_index = parts.index(".gist")
        if len(parts) > gist_index + 2 and parts[gist_index + 1] == "runs":
            return parts[gist_index + 2]
    compression = payload.get("compression", payload)
    return str(compression.get("video_id") or path.parent.parent.name)


def _proposal_rationale(
    category: QueryIntent,
    category_counts: Counter[str],
    needed_by_category: dict[str, int],
) -> str:
    needed = needed_by_category.get(category.value, 0)
    if needed:
        return (
            f"{category.value} has {category_counts[category.value]} curated case(s); "
            f"{needed} more needed for the configured gate."
        )
    return (
        f"{category.value} is among the least-covered categories and can help grow "
        "the long-video suite toward the total-case gate."
    )


def _artifact_source_path(payload: dict, artifact_path: Path) -> Path:
    ingestion = payload.get("ingestion")
    if not isinstance(ingestion, dict):
        raise ValueError(f"{artifact_path}: ingestion metadata is required")
    source = ingestion.get("source_path")
    if not source:
        raise ValueError(f"{artifact_path}: ingestion.source_path is required")
    source_path = Path(str(source)).expanduser()
    if source_path.is_absolute():
        return source_path
    if source_path.exists():
        return source_path.resolve(strict=False)
    return (artifact_path.parent / source_path).resolve(strict=False)


def _curation_review(
    *,
    proposal: LongVideoQueryProposal,
    draft: QualityCaseDraft,
    compression_payload,
) -> LongVideoCurationReview:
    case = draft.case
    warnings = [
        "Human review is required before appending this draft to the curated dataset."
    ]
    checklist = [
        "Open report.html and verify the answer is supported by the displayed video evidence.",
        "Replace inferred expected_answer_terms with terms from the verified answer.",
        "Replace inferred expected_evidence_terms with terms visible/audible in the verified evidence.",
        "Tighten relevant_ranges to the exact answer-supporting timestamps.",
        "Set min_grounded_evidence_rate above 0 after confirming evidence grounding.",
    ]
    if not case.relevant_ranges and not case.relevant_timestamps:
        warnings.append("Draft has no timestamp ranges; add verified answer ranges.")
    if case.min_grounded_evidence_rate == 0:
        warnings.append("Grounding threshold is not enforced yet.")
    if proposal.query_category == QueryIntent.MIXED_AV:
        if case.min_visual_evidence == 0:
            warnings.append("Mixed-AV draft does not require visual evidence yet.")
        if case.min_audio_evidence == 0:
            warnings.append("Mixed-AV draft does not require audio evidence yet.")
    if compression_payload.query_intent != proposal.query_category:
        warnings.append(
            "Generated compression intent "
            f"`{compression_payload.query_intent}` differs from proposal category "
            f"`{proposal.query_category}`."
        )
    warning_codes = [warning.code for warning in compression_payload.quality_warnings]
    if warning_codes:
        warnings.append(f"Runtime quality warnings present: {', '.join(warning_codes)}.")
    if _terms_look_noisy(case.expected_answer_terms + case.expected_evidence_terms):
        warnings.append("Drafted terms look noisy; replace OCR/transcript artifacts manually.")
    recommendation = (
        "Review and edit the draft before dataset append. "
        "Do not treat this as an accepted quality case until the checklist is complete."
    )
    return LongVideoCurationReview(
        ready_for_dataset=False,
        warnings=warnings,
        checklist=checklist,
        recommendation=recommendation,
    )


def _review_quality_case_for_dataset(case: QualityCase) -> LongVideoCurationReview:
    warnings: list[str] = []
    checklist: list[str] = []
    if case.compression_path is None:
        warnings.append("compression_path is required for curated long-video replay cases.")
    elif not case.compression_path.exists():
        warnings.append(f"compression_path does not exist: {case.compression_path}")
    if case.query_category is None:
        warnings.append("query_category is required.")
    if not case.domain or not case.domain.strip():
        warnings.append("domain is required.")
    if not case.expected_answer_terms:
        warnings.append("expected_answer_terms must be manually verified and non-empty.")
    if not case.expected_evidence_terms:
        warnings.append("expected_evidence_terms must be manually verified and non-empty.")
    if _terms_look_noisy(case.expected_answer_terms + case.expected_evidence_terms):
        warnings.append("Expected terms look noisy; replace OCR/transcript artifacts.")
    if not case.relevant_ranges and not case.relevant_timestamps:
        warnings.append("At least one relevant timestamp or range is required.")
    if case.min_answer_term_recall < 0.75:
        warnings.append("min_answer_term_recall should be at least 0.75.")
    if case.min_evidence_relevance_rate < 0.8:
        warnings.append("min_evidence_relevance_rate should be at least 0.80.")
    if case.min_timestamp_hit_rate < 0.75:
        warnings.append("min_timestamp_hit_rate should be at least 0.75.")
    if case.min_grounded_evidence_rate <= 0:
        warnings.append("min_grounded_evidence_rate must be enforced after review.")
    if case.min_token_reduction_percent < 90:
        warnings.append("min_token_reduction_percent should be at least 90 for long videos.")
    if case.max_selected_evidence is None:
        warnings.append("max_selected_evidence should be capped for curated cases.")
    if case.query_category == QueryIntent.MIXED_AV:
        if case.min_visual_evidence < 1:
            warnings.append("mixed_av cases should require at least one visual evidence item.")
        if case.min_audio_evidence < 1:
            warnings.append("mixed_av cases should require at least one audio/transcript evidence item.")

    if warnings:
        checklist = [
            "Open report.html and verify the answer-supporting evidence.",
            "Edit the draft JSONL until this review command returns ready_for_dataset=yes.",
            "Run gist-quality-eval against the edited draft before appending it.",
        ]
        recommendation = "Do not append this draft yet."
    else:
        checklist = [
            "Append the reviewed JSONL line to data/eval/long-video-quality.jsonl.",
            "Run gist-long-video-suite with --run-quality to confirm the curated suite still passes baseline gates.",
        ]
        recommendation = "Draft is structurally ready for dataset append."
    return LongVideoCurationReview(
        ready_for_dataset=not warnings,
        warnings=warnings,
        checklist=checklist,
        recommendation=recommendation,
    )


def _terms_look_noisy(terms: list[str]) -> bool:
    if not terms:
        return True
    noisy_terms = {"near", "seconds", "text", "screen", "visual", "most"}
    hits = sum(term.lower() in noisy_terms or len(term) <= 2 for term in terms)
    return hits / len(terms) >= 0.3


def _render_curation_review_markdown(review: LongVideoCurationReview) -> str:
    warnings = "\n".join(f"- {warning}" for warning in review.warnings)
    checklist = "\n".join(f"- [ ] {item}" for item in review.checklist)
    return f"""# Gist Curation Review

- Ready for dataset: {"yes" if review.ready_for_dataset else "no"}
- Recommendation: {review.recommendation}

## Warnings

{warnings or "- None"}

## Checklist

{checklist or "- None"}
"""


def _safe_stem(value: str | Path) -> str:
    text = Path(value).stem if isinstance(value, Path) else str(value)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-")
    return slug or "gist-run"


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _at_least(
    name: str,
    actual: float,
    required: float,
) -> LongVideoSuiteGateResult:
    passed = actual >= required
    return LongVideoSuiteGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} >= {required:.2f}" if passed else f"{actual:.2f} < {required:.2f}",
    )


def _at_most(
    name: str,
    actual: float,
    required: float,
) -> LongVideoSuiteGateResult:
    passed = actual <= required
    return LongVideoSuiteGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} <= {required:.2f}" if passed else f"{actual:.2f} > {required:.2f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
