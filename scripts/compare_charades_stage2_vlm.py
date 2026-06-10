from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from video_analyst.video_io import seconds_to_timestamp


CHARADES_TO_CANONICAL = {
    "c015": "holding_phone",
    "c016": "looking_at_phone",
    "c019": "talking_on_phone",
    "c051": "watching_laptop",
    "c052": "using_laptop",
    "c059": "sitting_on_chair",
    "c065": "eating",
    "c097": "walking_through_doorway",
    "c106": "drinking_water",
    "c107": "holding_drink_container",
    "c109": "putting_drink_container",
    "c110": "taking_drink_container",
    "c123": "sitting_on_sofa",
    "c125": "sitting_on_floor",
    "c132": "watching_tv",
    "c147": "cooking",
    "c150": "running",
    "c151": "sitting_down",
    "c154": "standing_up",
    "c156": "eating",
}


CANONICAL_ACTION_NAMES = {
    "drinking_water": "喝水",
    "talking_on_phone": "打电话",
    "holding_phone": "拿手机",
    "looking_at_phone": "看手机",
    "holding_drink_container": "拿杯/瓶",
    "taking_drink_container": "拿起杯/瓶",
    "putting_drink_container": "放下杯/瓶",
    "watching_laptop": "看笔记本电脑",
    "using_laptop": "使用笔记本电脑",
    "sitting_on_chair": "坐在椅子上",
    "sitting_on_sofa": "坐在沙发上",
    "sitting_on_floor": "坐在地上",
    "sitting_down": "坐下",
    "standing_up": "站起",
    "walking_through_doorway": "穿过门口",
    "running": "跑步",
    "watching_tv": "看电视",
    "cooking": "做饭",
    "eating": "吃东西",
}


@dataclass(frozen=True)
class CharadesEvidence:
    action_ids: list[str]
    action_names: list[str]
    intervals: list[dict[str, float]]


@dataclass(frozen=True)
class Stage2Evidence:
    present: bool
    events: list[dict[str, Any]]
    max_confidence: float


@dataclass(frozen=True)
class VlmEvidence:
    present: bool
    confidence: float
    evidence: str
    supporting_frame_indices: list[int]


@dataclass(frozen=True)
class ComparisonRow:
    action: str
    action_name: str
    charades_present: bool
    charades: CharadesEvidence
    stage2_present: bool
    stage2: Stage2Evidence
    vlm_present: bool
    vlm: VlmEvidence
    fused_status: str
    recommendation: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Charades labels, Stage2 events, and VLM review for one video.")
    parser.add_argument("--manifest", type=Path, required=True, help="Charades action manifest CSV.")
    parser.add_argument("--video-id", required=True, help="Charades video ID.")
    parser.add_argument("--stage2-dir", type=Path, required=True, help="Stage2 output directory.")
    parser.add_argument("--vlm-review", type=Path, required=True, help="VLM review JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Comparison JSON output path.")
    parser.add_argument(
        "--stage2-time-offset-seconds",
        type=float,
        default=0.0,
        help="Add this offset to Stage2 event times. Use clip_start_seconds when Stage2 ran on a sliced clip.",
    )
    args = parser.parse_args()

    rows = compare_video(
        args.manifest,
        args.video_id,
        args.stage2_dir,
        args.vlm_review,
        stage2_time_offset_seconds=args.stage2_time_offset_seconds,
    )
    summary = build_summary(args.video_id, rows)
    payload = {
        "video_id": args.video_id,
        "summary": summary,
        "comparisons": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare_video(
    manifest_path: Path,
    video_id: str,
    stage2_dir: Path,
    vlm_review_path: Path,
    stage2_time_offset_seconds: float = 0.0,
) -> list[ComparisonRow]:
    charades = load_charades_evidence(manifest_path, video_id)
    stage2 = load_stage2_evidence(stage2_dir / "events.json", time_offset_seconds=stage2_time_offset_seconds)
    vlm = load_vlm_evidence(vlm_review_path)
    actions = sorted(
        set(CANONICAL_ACTION_NAMES) | set(charades) | set(stage2) | set(vlm),
        key=lambda action: (action not in CANONICAL_ACTION_NAMES, action),
    )

    rows: list[ComparisonRow] = []
    for action in actions:
        charades_evidence = charades.get(action, CharadesEvidence([], [], []))
        stage2_evidence = stage2.get(action, Stage2Evidence(False, [], 0.0))
        vlm_evidence = vlm.get(action, VlmEvidence(False, 0.0, "", []))
        fused_status, recommendation = fuse_action(
            charades_present=bool(charades_evidence.action_ids),
            stage2_present=stage2_evidence.present,
            vlm_present=vlm_evidence.present,
            stage2_confidence=stage2_evidence.max_confidence,
            vlm_confidence=vlm_evidence.confidence,
        )
        rows.append(
            ComparisonRow(
                action=action,
                action_name=CANONICAL_ACTION_NAMES.get(action, action),
                charades_present=bool(charades_evidence.action_ids),
                charades=charades_evidence,
                stage2_present=stage2_evidence.present,
                stage2=stage2_evidence,
                vlm_present=vlm_evidence.present,
                vlm=vlm_evidence,
                fused_status=fused_status,
                recommendation=recommendation,
            )
        )
    return rows


def load_charades_evidence(manifest_path: Path, video_id: str) -> dict[str, CharadesEvidence]:
    grouped: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            if record["video_id"] != video_id:
                continue
            canonical = CHARADES_TO_CANONICAL.get(record["action_id"])
            if canonical is None:
                continue
            item = grouped.setdefault(canonical, {"action_ids": [], "action_names": [], "intervals": []})
            if record["action_id"] not in item["action_ids"]:
                item["action_ids"].append(record["action_id"])
            if record["action_name"] not in item["action_names"]:
                item["action_names"].append(record["action_name"])
            item["intervals"].append(
                {
                    "start_seconds": round(float(record["start_seconds"]), 4),
                    "end_seconds": round(float(record["end_seconds"]), 4),
                }
            )
    return {
        action: CharadesEvidence(
            action_ids=payload["action_ids"],
            action_names=payload["action_names"],
            intervals=payload["intervals"],
        )
        for action, payload in grouped.items()
    }


def load_stage2_evidence(events_path: Path, time_offset_seconds: float = 0.0) -> dict[str, Stage2Evidence]:
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, event in enumerate(events):
        normalized = normalize_stage2_event(event, index=index, time_offset_seconds=time_offset_seconds)
        grouped.setdefault(normalized["action"], []).append(normalized)
    return {
        action: Stage2Evidence(
            present=True,
            events=items,
            max_confidence=max(float(item.get("confidence", 0.0)) for item in items),
        )
        for action, items in grouped.items()
    }


def normalize_stage2_event(event: dict[str, Any], index: int, time_offset_seconds: float) -> dict[str, Any]:
    normalized = dict(event)
    normalized.setdefault("stage2_event_index", index)
    if time_offset_seconds == 0:
        return normalized

    clip_start = float(event.get("start_seconds", 0.0))
    clip_end = float(event.get("end_seconds", 0.0))
    source_start = round(clip_start + time_offset_seconds, 4)
    source_end = round(clip_end + time_offset_seconds, 4)
    normalized["clip_start_seconds"] = clip_start
    normalized["clip_end_seconds"] = clip_end
    normalized["clip_start_time"] = event.get("start_time")
    normalized["clip_end_time"] = event.get("end_time")
    normalized["time_offset_seconds"] = round(time_offset_seconds, 4)
    normalized["start_seconds"] = source_start
    normalized["end_seconds"] = source_end
    normalized["start_time"] = seconds_to_timestamp(source_start)
    normalized["end_time"] = seconds_to_timestamp(source_end)
    normalized["time_coordinate"] = "source_video"
    return normalized


def load_vlm_evidence(vlm_review_path: Path) -> dict[str, VlmEvidence]:
    payload = json.loads(vlm_review_path.read_text(encoding="utf-8"))
    evidence: dict[str, VlmEvidence] = {}
    for item in payload.get("actions", []):
        action = canonicalize_action(item["action"])
        evidence[action] = VlmEvidence(
            present=bool(item.get("present", False)),
            confidence=float(item.get("confidence", 0.0)),
            evidence=str(item.get("evidence", "")),
            supporting_frame_indices=[int(index) for index in item.get("supporting_frame_indices", [])],
        )
    return evidence


def canonicalize_action(action: str) -> str:
    return CHARADES_TO_CANONICAL.get(action, action)


def fuse_action(
    charades_present: bool,
    stage2_present: bool,
    vlm_present: bool,
    stage2_confidence: float,
    vlm_confidence: float,
) -> tuple[str, str]:
    if stage2_present and vlm_present:
        return "confirmed_event", "保留事件，Stage2 和 VLM 一致。"
    if stage2_present and not vlm_present:
        if charades_present:
            return "needs_temporal_review", "Charades 有标注但 VLM 关键帧未确认，建议围绕 Stage2 事件加密抽帧复核。"
        return "possible_false_positive", "Stage2 命中但 VLM 否定，建议降置信或进入人工复核。"
    if not stage2_present and vlm_present:
        if charades_present:
            return "vlm_recovered_label", "Stage2 漏检但 VLM 与 Charades 一致，可生成语义候选事件。"
        return "vlm_candidate_event", "VLM 高置信命中但无 Charades/Stage2 支持，建议作为候选事件。"
    if charades_present:
        return "missed_label", "Charades 有标注，但 Stage2 和 VLM 均未确认，建议改进抽帧或检测模型。"
    return "not_present", "三方均未支持该动作。"


def build_summary(video_id: str, rows: list[ComparisonRow]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.fused_status] = status_counts.get(row.fused_status, 0) + 1
    return {
        "video_id": video_id,
        "num_actions": len(rows),
        "status_counts": status_counts,
        "stage2_hits": sum(1 for row in rows if row.stage2_present),
        "vlm_hits": sum(1 for row in rows if row.vlm_present),
        "charades_labels": sum(1 for row in rows if row.charades_present),
    }


if __name__ == "__main__":
    main()
