from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODELS_URL = "https://api.siliconflow.cn/v1/models"
VISION_PATTERNS = [
    r"\bvl\b",
    r"vision",
    r"visual",
    r"glm-?5v",
    r"qwen.*vl",
    r"internvl",
    r"llava",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="List SiliconFlow models and optionally filter likely vision-capable models.")
    parser.add_argument("--models-url", default=DEFAULT_MODELS_URL, help="SiliconFlow models endpoint.")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY", help="Environment variable that stores the API key.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional dotenv file to load before reading the API key.")
    parser.add_argument("--vision-only", action="store_true", help="Only print likely vision-capable models.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}.")

    try:
        response = fetch_models(args.models_url, api_key)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "http_error",
                    "status_code": exc.code,
                    "reason": exc.reason,
                    "body": error_body,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except urllib.error.URLError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "url_error",
                    "reason": str(exc.reason),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    models = extract_model_ids(response)
    if args.vision_only:
        models = filter_vision_models(models)

    result = {
        "models_url": args.models_url,
        "vision_only": args.vision_only,
        "count": len(models),
        "models": models,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def fetch_models(models_url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_model_ids(response: dict[str, Any]) -> list[str]:
    data = response.get("data", response.get("models", []))
    model_ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                model_ids.append(item)
            elif isinstance(item, dict):
                model_id = item.get("id") or item.get("name") or item.get("model")
                if model_id:
                    model_ids.append(str(model_id))
    return sorted(dict.fromkeys(model_ids))


def filter_vision_models(model_ids: list[str]) -> list[str]:
    patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in VISION_PATTERNS]
    return [model_id for model_id in model_ids if any(pattern.search(model_id) for pattern in patterns)]


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
