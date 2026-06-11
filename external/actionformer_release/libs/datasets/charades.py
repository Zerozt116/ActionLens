import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("charades")
class CharadesClipDataset(Dataset):
    """Charades clip-level adapter for ActionFormer.

    The project keeps Charades annotations at full-video granularity, while our
    pilot pipeline slices clips around selected actions. This loader intersects
    each full-video annotation with its clip window and returns clip-local
    segments in the same C x T feature contract used by the official loaders.
    """

    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        json_file,
        clips_manifest,
        feature_manifest,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        num_classes,
        file_prefix,
        file_ext,
        force_upsampling,
        split_folder=None,
    ):
        assert os.path.exists(json_file)
        assert os.path.exists(clips_manifest)
        assert os.path.exists(feature_manifest)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        if feat_folder is not None:
            assert os.path.exists(feat_folder)

        self.is_training = is_training
        self.split = split
        self.feat_folder = feat_folder
        self.json_file = json_file
        self.clips_manifest = clips_manifest
        self.feature_manifest = feature_manifest
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.file_prefix = file_prefix or ""
        self.file_ext = file_ext or ".npz"
        self.force_upsampling = force_upsampling
        self.split_folder = split_folder

        self.data_list, self.label_dict = self._load_db()
        self.eval_json_file = None
        if len(self.label_dict) > num_classes:
            raise ValueError(
                f"Charades label_dict has {len(self.label_dict)} classes, "
                f"but config num_classes={num_classes}"
            )

        used_label_ids = set(int(value) for value in self.label_dict.values())
        self.db_attributes = {
            "dataset_name": "charades",
            "tiou_thresholds": np.linspace(0.1, 0.5, 5),
            "empty_label_ids": [
                idx for idx in range(num_classes) if idx not in used_label_ids
            ],
        }

    def get_attributes(self):
        return self.db_attributes

    def get_eval_json_file(self):
        """Write clip-level GT JSON for ActionFormer/ActivityNet evaluation.

        The dataset returns clip IDs as prediction video IDs. The source
        annotation JSON is keyed by full Charades video IDs, so evaluation needs
        a derived ground-truth file keyed by the same clip IDs that inference
        emits.
        """
        if self.eval_json_file is not None and os.path.exists(self.eval_json_file):
            return self.eval_json_file

        split_name = "_".join(str(value).lower() for value in self.split) or "all"
        output_dir = Path(self.clips_manifest).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"charades_clip_eval_{split_name}.json"
        id_to_label = {int(value): key for key, value in self.label_dict.items()}
        database = {}
        for item in self.data_list:
            annotations = []
            for segment, label_id in zip(item["segments"], item["labels"]):
                label_id = int(label_id)
                annotations.append(
                    {
                        "segment": [
                            round(float(segment[0]), 4),
                            round(float(segment[1]), 4),
                        ],
                        "label_id": label_id,
                        "label": id_to_label.get(label_id, str(label_id)),
                    }
                )
            database[item["id"]] = {
                "subset": split_name,
                "duration": round(float(item["duration"]), 4),
                "fps": round(float(item["fps"]), 4),
                "annotations": annotations,
            }

        output_path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "source_json_file": self.json_file,
                    "source_clips_manifest": self.clips_manifest,
                    "label_dict": self.label_dict,
                    "database": database,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.eval_json_file = str(output_path)
        return self.eval_json_file

    def _load_db(self):
        with open(self.json_file, "r", encoding="utf-8") as file:
            annotations_payload = json.load(file)
        annotation_db = annotations_payload.get("database", {})
        label_dict = annotations_payload.get("label_dict") or build_label_dict(annotation_db)

        with open(self.feature_manifest, "r", encoding="utf-8") as file:
            feature_payload = json.load(file)
        feature_db = feature_payload.get("database", {})

        allowed_video_ids = read_split_video_ids(self.split, self.split_folder)
        rows = read_clip_rows(Path(self.clips_manifest))
        data_list = []
        for row in rows:
            clip_id = row["clip_id"]
            feature_entry = feature_db.get(clip_id)
            if not feature_entry:
                continue
            feature_path = resolve_feature_path(
                feature_entry=feature_entry,
                feat_folder=self.feat_folder,
                clip_id=clip_id,
                file_prefix=self.file_prefix,
                file_ext=self.file_ext,
            )
            if not feature_path.exists():
                continue

            video_id = row["video_id"]
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            clip_start = parse_float(row.get("clip_start_seconds"), 0.0)
            clip_end = parse_float(row.get("clip_end_seconds"), None)
            if clip_end is None:
                duration = parse_float(feature_entry.get("duration_seconds"), 0.0)
                clip_end = clip_start + duration
            annotations = annotation_db.get(video_id, {}).get("annotations", [])
            segments, labels = intersect_clip_annotations(
                annotations,
                clip_start=clip_start,
                clip_end=clip_end,
            )
            fps = self.default_fps
            if fps is None:
                fps = parse_float(feature_entry.get("source_fps"), None)
            if fps is None:
                fps = parse_float(annotation_db.get(video_id, {}).get("fps"), None)
            if fps is None:
                raise ValueError(f"Unknown FPS for clip {clip_id}")

            duration = parse_float(
                feature_entry.get("duration_seconds"),
                max(clip_end - clip_start, 0.0),
            )
            data_list.append(
                {
                    "id": clip_id,
                    "video_id": video_id,
                    "feature_path": str(feature_path),
                    "fps": float(fps),
                    "duration": float(duration),
                    "segments": (
                        np.asarray(segments, dtype=np.float32)
                        if segments
                        else np.empty((0, 2), dtype=np.float32)
                    ),
                    "labels": (
                        np.asarray(labels, dtype=np.int64)
                        if labels
                        else np.empty((0,), dtype=np.int64)
                    ),
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                }
            )

        return tuple(data_list), label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        feats = load_features(video_item["feature_path"], self.file_ext)
        if feats.ndim != 2:
            raise ValueError(
                f"Expected features T x C for {video_item['id']}, got shape {feats.shape}"
            )
        if feats.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim} for {video_item['id']}, "
                f"got {feats.shape[1]}"
            )

        feats = feats[:: self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        segments = torch.from_numpy(
            video_item["segments"] * video_item["fps"] / feat_stride - feat_offset
        )
        labels = torch.from_numpy(video_item["labels"])

        data_dict = {
            "video_id": video_item["id"],
            "feats": feats,
            "segments": segments,
            "labels": labels,
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": self.num_frames,
        }

        if self.is_training:
            data_dict = truncate_feats(
                data_dict,
                self.max_seq_len,
                self.trunc_thresh,
                feat_offset,
                self.crop_ratio,
                has_action=(labels.numel() > 0),
            )

        return data_dict


def build_label_dict(annotation_db):
    label_dict = {}
    for item in annotation_db.values():
        for annotation in item.get("annotations", []):
            label_dict[annotation["label"]] = int(annotation["label_id"])
    return label_dict


def read_clip_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") not in {"sliced", "exists"}:
                continue
            rows.append(row)
    return rows


def read_split_video_ids(split, split_folder):
    split_values = [str(value).lower() for value in split]
    if not split_values or "all" in split_values:
        return None
    if split_folder is None:
        return None
    folder = Path(split_folder)
    video_ids = set()
    for split_name in split_values:
        path = folder / f"{split_name}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                video_ids.add(value)
    return video_ids if video_ids else None


def resolve_feature_path(feature_entry, feat_folder, clip_id, file_prefix, file_ext):
    feature_path = feature_entry.get("feature_path")
    if feature_path:
        return Path(feature_path)
    return Path(feat_folder) / f"{file_prefix}{clip_id}{file_ext}"


def parse_float(value, default):
    if value is None or value == "":
        return default
    return float(value)


def intersect_clip_annotations(annotations, clip_start, clip_end):
    segments = []
    labels = []
    for annotation in annotations:
        start, end = annotation.get("segment", [0.0, 0.0])
        left = max(float(start), float(clip_start))
        right = min(float(end), float(clip_end))
        if right <= left:
            continue
        segments.append([left - clip_start, right - clip_start])
        labels.append(int(annotation["label_id"]))
    return segments, labels


def load_features(path, file_ext):
    if file_ext == ".npz":
        with np.load(path) as data:
            return data["feats"].astype(np.float32)
    return np.load(path).astype(np.float32)
