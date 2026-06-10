from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from video_analyst.video_io import probe_video


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_ACTIONS = ["drinking_water", "talking_on_phone", "holding_phone", "looking_at_phone", "eating"]
ACTION_DEFINITIONS = {
    "drinking_water": "A person drinks any beverage from a cup, bottle, glass, can, or similar container. It does not need to be literally water.",
    "talking_on_phone": "A person holds a phone near the ear or mouth and appears to be making a phone call.",
    "holding_phone": "A person visibly holds a phone or camera-like phone device.",
    "looking_at_phone": "A person looks at a phone screen or appears to use a phone visually.",
    "eating": "A person eats food or brings food to the mouth.",
    "watching_laptop": "A person looks at a laptop screen without necessarily typing or touching it.",
    "using_laptop": "A person interacts with a laptop, such as typing, using the trackpad, or working/playing on it.",
    "sitting_on_chair": "A person is seated on a chair.",
    "sitting_on_sofa": "A person is seated on a sofa or couch.",
    "sitting_on_floor": "A person is seated on the floor.",
    "sitting_down": "A person transitions from standing to sitting.",
    "standing_up": "A person transitions from sitting, lying, or crouching to standing.",
    "walking_through_doorway": "A person walks through or across a doorway.",
    "running": "A person runs or moves quickly with a running gait.",
    "watching_tv": "A person looks at or watches a television.",
    "cooking": "A person prepares or cooks food, usually near kitchen equipment or food items.",
    "holding_drink_container": "A person holds a cup, glass, bottle, can, or similar drink container.",
    "taking_drink_container": "A person picks up or takes a cup, glass, bottle, can, or similar drink container.",
    "putting_drink_container": "A person puts down or places a cup, glass, bottle, can, or similar drink container.",
}


@dataclass(frozen=True)
class ExtractedFrame:
    index: int
    timestamp_seconds: float
    frame_number: int
    path: str


@dataclass(frozen=True)
class ReviewWindow:
    start_seconds: float | None
    end_seconds: float | None
    source: str
    event: dict[str, Any] | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a video segment with a multimodal model using sampled key frames.")
    parser.add_argument("video", type=Path, help="Video path.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for frames, payload, and response.")
    parser.add_argument("--provider", default="siliconflow", choices=["siliconflow"], help="VLM provider.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Chat completions endpoint.")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY", help="Environment variable that stores the API key.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional dotenv file to load before reading the API key.")
    parser.add_argument("--frame-count", type=int, default=6, help="Number of key frames to sample.")
    parser.add_argument("--start", type=float, default=None, help="Optional segment start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="Optional segment end time in seconds.")
    parser.add_argument("--actions", nargs="+", default=None, help="Actions for the VLM to review.")
    parser.add_argument("--events-json", type=Path, default=None, help="Optional Stage2 events.json path for event-centered review.")
    parser.add_argument("--event-index", type=int, default=None, help="Zero-based event index from --events-json.")
    parser.add_argument("--event-context-seconds", type=float, default=1.0, help="Context seconds before and after the selected event.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for extracted frames.")
    parser.add_argument("--dry-run", action="store_true", help="Only extract frames and write the request payload.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(args.video)
    review_window = resolve_review_window(
        explicit_start=args.start,
        explicit_end=args.end,
        events_json=args.events_json,
        event_index=args.event_index,
        event_context_seconds=args.event_context_seconds,
        video_duration_seconds=metadata.duration_seconds,
    )
    actions = resolve_actions(args.actions, review_window.event)
    frames_dir = args.output_dir / "frames"
    frames = extract_key_frames(
        video_path=args.video,
        output_dir=frames_dir,
        frame_count=args.frame_count,
        start_seconds=review_window.start_seconds,
        end_seconds=review_window.end_seconds,
        jpeg_quality=args.jpeg_quality,
    )
    payload = build_payload(
        model=args.model,
        video_path=args.video,
        metadata=metadata.to_dict(),
        frames=frames,
        actions=actions,
        review_context=asdict(review_window),
    )
    payload_path = args.output_dir / "vlm_request_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "video": str(args.video),
        "frames": [asdict(frame) for frame in frames],
        "review_window": asdict(review_window),
        "actions": actions,
        "request_payload": str(payload_path),
        "dry_run": args.dry_run,
    }

    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    if args.dry_run or not api_key:
        result["status"] = "dry_run" if args.dry_run else "missing_api_key"
        result["message"] = f"Set {args.api_key_env} to call the VLM API." if not api_key else "Dry run requested."
    else:
        try:
            response = call_chat_completions(args.base_url, api_key, payload)
            raw_path = args.output_dir / "vlm_response_raw.json"
            raw_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
            review = parse_review_response(response)
            review_path = args.output_dir / "vlm_review.json"
            review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
            result.update(
                {
                    "status": "completed",
                    "raw_response": str(raw_path),
                    "review": str(review_path),
                    "review_json": review,
                }
            )
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            result.update(
                {
                    "status": "failed",
                    "error": "http_error",
                    "status_code": exc.code,
                    "reason": exc.reason,
                    "body": error_body,
                }
            )
        except urllib.error.URLError as exc:
            result.update(
                {
                    "status": "failed",
                    "error": "url_error",
                    "reason": str(exc.reason),
                }
            )

    summary_path = args.output_dir / "vlm_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


def resolve_review_window(
    explicit_start: float | None,
    explicit_end: float | None,
    events_json: Path | None,
    event_index: int | None,
    event_context_seconds: float,
    video_duration_seconds: float,
) -> ReviewWindow:
    if events_json is None:
        return ReviewWindow(explicit_start, explicit_end, "explicit_or_full_video", None)
    if event_index is None:
        raise ValueError("--event-index is required when --events-json is provided")
    events = json.loads(events_json.read_text(encoding="utf-8"))
    if event_index < 0 or event_index >= len(events):
        raise IndexError(f"event-index {event_index} out of range for {events_json}")
    event = events[event_index]
    start = max(0.0, float(event["start_seconds"]) - event_context_seconds)
    end = min(video_duration_seconds, float(event["end_seconds"]) + event_context_seconds)
    return ReviewWindow(round(start, 4), round(end, 4), f"event:{event_index}", event)


def resolve_actions(actions: list[str] | None, event: dict[str, Any] | None) -> list[str]:
    if actions:
        return actions
    if event is not None and event.get("action"):
        return [str(event["action"])]
    return DEFAULT_ACTIONS


def extract_key_frames(
    video_path: Path,
    output_dir: Path,
    frame_count: int,
    start_seconds: float | None,
    end_seconds: float | None,
    jpeg_quality: int,
) -> list[ExtractedFrame]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    metadata = probe_video(video_path)
    start = 0.0 if start_seconds is None else max(0.0, start_seconds)
    end = metadata.duration_seconds if end_seconds is None else min(metadata.duration_seconds, end_seconds)
    if end <= start:
        raise ValueError(f"Invalid review window: start={start}, end={end}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = sample_timestamps(start, end, frame_count)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames: list[ExtractedFrame] = []
    try:
        for index, timestamp in enumerate(timestamps):
            frame_number = max(0, min(metadata.frame_count - 1, int(round(timestamp * metadata.fps)))) if metadata.fps > 0 else 0
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            output_path = output_dir / f"frame_{index:02d}_{timestamp:.3f}s.jpg"
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frames.append(
                ExtractedFrame(
                    index=index,
                    timestamp_seconds=round(timestamp, 4),
                    frame_number=frame_number,
                    path=str(output_path),
                )
            )
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    return frames


def sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (count + 1)
    return [start + step * (index + 1) for index in range(count)]


def build_payload(
    model: str,
    video_path: Path,
    metadata: dict[str, Any],
    frames: list[ExtractedFrame],
    actions: list[str],
    review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_blocks: list[dict[str, Any]] = []
    for frame in frames:
        mime_type = "image/jpeg"
        data_url = f"data:{mime_type};base64,{base64.b64encode(Path(frame.path).read_bytes()).decode('ascii')}"
        frame_blocks.append({"type": "text", "text": f"Frame {frame.index}: timestamp={frame.timestamp_seconds:.3f}s frame_number={frame.frame_number}"})
        frame_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    prompt = build_prompt(video_path, metadata, actions, review_context=review_context)
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *frame_blocks],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }


def build_prompt(video_path: Path, metadata: dict[str, Any], actions: list[str], review_context: dict[str, Any] | None = None) -> str:
    action_list = ", ".join(actions)
    action_definitions = {action: ACTION_DEFINITIONS[action] for action in actions if action in ACTION_DEFINITIONS}
    context_text = f"Review context: {json.dumps(review_context, ensure_ascii=False)}\n" if review_context else ""
    definitions_text = f"Action definitions: {json.dumps(action_definitions, ensure_ascii=False)}\n" if action_definitions else ""
    return (
        "You are reviewing sampled key frames from a video for human activity analysis.\n"
        "Return strict JSON only. Do not include markdown.\n"
        f"Video: {video_path}\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"{context_text}"
        f"Actions to review: {action_list}\n"
        f"{definitions_text}"
        "For each action, decide whether the visible person is performing it in the sampled frames.\n"
        "If review context includes a Stage2 event, specifically confirm or reject that event using the sampled frames.\n"
        "Use the frame timestamps as evidence. If a phone or cup is not clearly visible, say so.\n"
        "JSON schema:\n"
        "{\n"
        '  "overall_summary": "short summary",\n'
        '  "actions": [\n'
        "    {\n"
        '      "action": "drinking_water",\n'
        '      "present": true,\n'
        '      "confidence": 0.0,\n'
        '      "evidence": "what is visible",\n'
        '      "supporting_frame_indices": [0]\n'
        "    }\n"
        "  ],\n"
        '  "visible_objects": ["cup", "phone"],\n'
        '  "risk_notes": ["possible occlusion"]\n'
        "}\n"
    )


def call_chat_completions(base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_review_response(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    else:
        text = str(content)
    text = strip_code_fence(text).strip()
    return json.loads(text)


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
