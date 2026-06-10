from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.list_siliconflow_models import extract_model_ids, filter_vision_models, load_env_file


class ListSiliconFlowModelsTests(unittest.TestCase):
    def test_extract_model_ids_from_openai_style_response(self) -> None:
        response = {
            "data": [
                {"id": "Qwen/Qwen3-VL-8B-Instruct"},
                {"id": "deepseek-ai/DeepSeek-V3"},
                {"name": "GLM-5V-Turbo"},
                "custom-model",
            ]
        }

        self.assertEqual(
            extract_model_ids(response),
            ["GLM-5V-Turbo", "Qwen/Qwen3-VL-8B-Instruct", "custom-model", "deepseek-ai/DeepSeek-V3"],
        )

    def test_filter_vision_models(self) -> None:
        models = [
            "Qwen/Qwen3-VL-8B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "GLM-5V-Turbo",
            "OpenGVLab/InternVL3-8B",
            "llava-hf/llava-1.5-7b",
        ]

        self.assertEqual(
            filter_vision_models(models),
            ["Qwen/Qwen3-VL-8B-Instruct", "GLM-5V-Turbo", "OpenGVLab/InternVL3-8B", "llava-hf/llava-1.5-7b"],
        )

    def test_load_env_file_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("SILICONFLOW_API_KEY=file-key\nOTHER=value\n", encoding="utf-8")
            original = os.environ.get("SILICONFLOW_API_KEY")
            os.environ["SILICONFLOW_API_KEY"] = "existing-key"
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["SILICONFLOW_API_KEY"], "existing-key")
                self.assertEqual(os.environ["OTHER"], "value")
            finally:
                if original is None:
                    os.environ.pop("SILICONFLOW_API_KEY", None)
                else:
                    os.environ["SILICONFLOW_API_KEY"] = original
                os.environ.pop("OTHER", None)


if __name__ == "__main__":
    unittest.main()
