"""Reconstructed strict prediction contract. No label-column inference."""
from dataclasses import dataclass, field
from pathlib import Path
import csv
import numpy as np

MODELS = ("Xception", "InceptionV3", "MobileNetV1")


def validate_mapping(mapping):
    if set(mapping) != {"real", "fake"} or set(mapping.values()) != {0, 1}:
        raise ValueError("Explicit mapping must give distinct real/fake indices0,1")
    return dict(mapping)


def validate_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA256")
    return value


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected nonempty finite [0,1] fake probabilities")
    return p


def fake_column(values, mapping):
    mapping = validate_mapping(mapping)
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("Expected Nx2 probabilities")
    if np.any((a < 0) | (a > 1)) or not np.allclose(a.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Invalid probability distribution")
    return probabilities(a[:, mapping["fake"]])


def canonical_labels(labels, mapping):
    mapping = validate_mapping(mapping)
    a = np.asarray(labels)
    if a.ndim != 1 or not np.isin(a, [0, 1]).all():
        raise ValueError("Labels must be observed binary numeric class indices; missing labels forbidden")
    return (a == mapping["fake"]).astype(np.int64)


@dataclass(frozen=True)
class PredictionSet:
    model_name: str
    dataset_id: str
    split: str
    manifest_sha256: str
    checkpoint_sha256: str
    class_mapping: dict
    sample_ids: tuple
    p_fake: np.ndarray
    labels: object = None
    video_ids: object = None
    provenance: str = "unverified_existing_predictions"
    original_class_mapping: dict = field(init=False)

    def __post_init__(self):
        if not self.dataset_id or self.split not in {"train", "validation", "test"}:
            raise ValueError("Explicit dataset and train/validation/test split required")
        validate_sha(self.manifest_sha256); validate_sha(self.checkpoint_sha256)
        mapping = validate_mapping(self.class_mapping)
        ids = tuple(self.sample_ids)
        p = probabilities(self.p_fake).copy()
        if len(ids) != len(p) or len(set(ids)) != len(ids) or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError("Unique nonempty sample IDs required, aligned with probabilities")
        y = None if self.labels is None else canonical_labels(self.labels, mapping)
        if y is not None and len(y) != len(p):
            raise ValueError("Label length mismatch")
        videos = None if self.video_ids is None else tuple(self.video_ids)
        if videos is not None and (len(videos) != len(p) or any(not v for v in videos)):
            raise ValueError("Missing/misaligned video IDs")
        p.setflags(write=False)
        if y is not None:y.setflags(write=False)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "p_fake", p)
        object.__setattr__(self, "labels", y)
        object.__setattr__(self, "original_class_mapping", mapping)
        object.__setattr__(self, "class_mapping", {"real":0,"fake":1})
        object.__setattr__(self, "video_ids", videos)


def aligned(predictions, require_labels=False):
    items = tuple(predictions)
    if not items:
        raise ValueError("Predictions missing")
    first = items[0]
    for item in items:
        if (item.sample_ids != first.sample_ids or item.dataset_id != first.dataset_id or
            item.split != first.split or item.manifest_sha256 != first.manifest_sha256):
            raise ValueError("Dataset/split/manifest/sample ordering mismatch; no silent realignment")
        if item.video_ids != first.video_ids:
            raise ValueError("Video alignment mismatch")
        if require_labels and item.labels is None:
            raise ValueError("Genuine observed labels required")
        if (item.labels is None) != (first.labels is None) or (item.labels is not None and not np.array_equal(item.labels,first.labels)):
            raise ValueError("Ground-truth alignment mismatch")
    return items


def read_prediction_csv(path, *, model_name, dataset_id, split, manifest_sha256,
                        checkpoint_sha256, class_mapping, provenance="unverified_existing_predictions"):
    """CSV: sample_id,label(optional numeric),video_id(optional), p0,p1 OR p_fake."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:raise ValueError("Empty predictions")
    mapping = validate_mapping(class_mapping)
    ids = tuple(r["sample_id"] for r in rows)
    has_y = ["label" in r and r["label"] != "" for r in rows]
    if any(has_y) and not all(has_y):raise ValueError("Partially missing labels")
    labels = [int(r["label"]) for r in rows] if all(has_y) else None
    has_video = [bool(r.get("video_id")) for r in rows]
    if any(has_video) and not all(has_video):raise ValueError("Partially missing video IDs")
    video = tuple(r["video_id"] for r in rows) if all(has_video) else None
    if "p0" in rows[0] or "p1" in rows[0]:
        p = fake_column([[float(r["p0"]),float(r["p1"])] for r in rows],mapping)
        if "p_fake" in rows[0] and not np.allclose(p,[float(r["p_fake"]) for r in rows],atol=1e-8,rtol=0):
            raise ValueError("Conflicting fake column")
    else:p = [float(r["p_fake"]) for r in rows]
    return PredictionSet(model_name,dataset_id,split,manifest_sha256,checkpoint_sha256,
                         mapping,ids,p,labels,video,provenance)
