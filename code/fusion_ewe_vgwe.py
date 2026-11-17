"""Corrected EWE/VGWE, not authenticated historical fusion code."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json
import numpy as np
from prediction_data import MODELS,aligned,validate_sha
from evaluate_intradataset import binary_metrics


def _three(predictions, require_labels=False):
    if predictions is None:raise ValueError("Required source predictions are missing")
    items=aligned(predictions,require_labels=require_labels)
    if tuple(i.model_name for i in items)!=MODELS:
        raise ValueError("Require fixed order Xception, InceptionV3, MobileNetV1; no test selection")
    if len({i.checkpoint_sha256 for i in items})!=3:
        raise ValueError("Distinct branch checkpoint identities required")
    return items


def equal_weight(predictions):
    items=_three(predictions)
    return np.mean(np.stack([i.p_fake for i in items]),axis=0)


@dataclass(frozen=True)
class ValidationWeights:
    model_names: tuple
    weights: tuple
    validation_metrics: tuple
    dataset_id: str
    manifest_sha256: str
    checkpoint_sha256s: tuple
    validation_sample_ids: tuple
    prediction_provenance: tuple
    validation_video_ids: object

    def __post_init__(self):
        w=np.asarray(self.weights,dtype=float)
        if (tuple(self.model_names)!=MODELS or w.shape!=(3,) or not np.isfinite(w).all()
            or np.any(w<0) or not np.isclose(w.sum(),1,atol=1e-12,rtol=0)):
            raise ValueError("Invalid weights/model order")
        validate_sha(self.manifest_sha256)
        if not self.dataset_id or len(self.checkpoint_sha256s)!=3 or not self.validation_sample_ids:
            raise ValueError("Validation provenance/checkpoints missing")
        for h in self.checkpoint_sha256s:validate_sha(h)
        if len(self.validation_metrics)!=3:
            raise ValueError("Validation metrics missing")
        scores=np.array([(m["accuracy"]+m["f1"]+m["roc_auc"])/3 for m in self.validation_metrics])
        if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1) or scores.sum()<=0:
            raise ValueError("Invalid validation composites")
        if not np.allclose(w,scores/scores.sum(),atol=1e-12,rtol=0):
            raise ValueError("Weights do not match recorded validation composites")
        object.__setattr__(self,"weights",tuple(float(x) for x in w))
        object.__setattr__(self,"model_names",tuple(self.model_names))
        object.__setattr__(self,"checkpoint_sha256s",tuple(self.checkpoint_sha256s))
        object.__setattr__(self,"validation_sample_ids",tuple(self.validation_sample_ids))

    def save(self,path):
        with Path(path).open("x") as f:json.dump(asdict(self),f,indent=2,allow_nan=False)


def fit_validation_weights(predictions):
    items=_three(predictions,require_labels=True)
    if any(i.split!="validation" for i in items):
        raise ValueError("Weights may ONLY be fitted from source-validation predictions")
    metrics=tuple(binary_metrics(i.labels,i.p_fake) for i in items)
    scores=[]
    for m in metrics:
        vals=[m["accuracy"],m["f1"],m["roc_auc"]]
        if any(v is None or not np.isfinite(v) for v in vals):
            raise ValueError("Validation ACC/F1/AUC all must be defined")
        scores.append(sum(vals)/3)
    total=sum(scores)
    if total<=0:raise ValueError("Zero validation composite score")
    return ValidationWeights(MODELS,tuple(s/total for s in scores),metrics,items[0].dataset_id,
                             items[0].manifest_sha256,tuple(i.checkpoint_sha256 for i in items),
                             items[0].sample_ids,tuple(i.provenance for i in items),items[0].video_ids)


def weighted_predict(predictions, fitted):
    if not isinstance(fitted,ValidationWeights):raise ValueError("Source-validation fitted weights required")
    items=_three(predictions)
    if (items[0].dataset_id!=fitted.dataset_id or
        items[0].manifest_sha256!=fitted.manifest_sha256 or
        tuple(i.checkpoint_sha256 for i in items)!=fitted.checkpoint_sha256s):
        raise ValueError("Source dataset/split manifest/checkpoint identities changed")
    if items[0].split=="test" and set(items[0].sample_ids)&set(fitted.validation_sample_ids):
        raise ValueError("Test IDs overlap validation used for fusion fitting")
    if (items[0].split=="test" and fitted.validation_video_ids is not None
        and items[0].video_ids is not None
        and set(items[0].video_ids)&set(fitted.validation_video_ids)):
        raise ValueError("Test video IDs overlap validation videos")
    # Test labels never determine weights or fused scores.
    return np.tensordot(np.asarray(fitted.weights),np.stack([i.p_fake for i in items]),axes=1)


def load_and_verify_weights(path, validation_predictions):
    """Revalidate stored weights against supplied original source-validation data."""
    expected=fit_validation_weights(validation_predictions)
    recorded=json.loads(Path(path).read_text())
    canonical=json.loads(json.dumps(asdict(expected),allow_nan=False))
    if recorded!=canonical:
        raise ValueError("Stored weights/provenance differ from supplied source validation")
    return expected
