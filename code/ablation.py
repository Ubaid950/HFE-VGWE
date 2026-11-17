"""Reconstructed Table5 experiment plan; no training on import or planning."""
from dataclasses import dataclass,asdict
import hashlib,json
from prediction_data import validate_sha


@dataclass(frozen=True)
class FeatureConfig:
    dwt: bool
    sgfe: bool
    se: bool

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise ValueError("Feature flags must be explicit booleans")


# Actual six conditions in the locally approved PeerJ Table5.
CONDITIONS = (
    ("rgb",FeatureConfig(False,False,False)),
    ("rgb_db2",FeatureConfig(True,False,False)),
    ("rgb_sgfe",FeatureConfig(False,True,False)),
    ("rgb_se",FeatureConfig(False,False,True)),
    ("rgb_db2_sgfe",FeatureConfig(True,True,False)),
    ("full_hfe",FeatureConfig(True,True,True)),
)


def ablation_plan(manifest_sha256):
    validate_sha(manifest_sha256)
    return tuple({"condition":name,"backbone":"Xception","features":asdict(flags),
                  "manifest_sha256":manifest_sha256,
                  "checkpoint_namespace":"ablation/"+name,
                  "status":"planned_not_executed",
                  "classification":"reconstructed_corrected_architecture",
                  "test_selection":False} for name,flags in CONDITIONS)


def select_by_validation(records):
    if not records or any(r.get("split")!="validation" for r in records):
        raise ValueError("Selection requires validation-only records")
    if any(not isinstance(r.get("accuracy"),(float,int)) or not 0<=r["accuracy"]<=1 for r in records):
        raise ValueError("Defined validation accuracy required")
    # Selection is meaningful only within one declared dataset/split/backbone.
    # Manifest hashes identify supplied records; they do not authenticate lineage.
    signatures=set()
    for record in records:
        dataset=record.get("dataset_id")
        backbone=record.get("backbone")
        if not isinstance(dataset,str) or not dataset.strip():
            raise ValueError("Explicit dataset_id required for comparable ablations")
        if not isinstance(backbone,str) or not backbone.strip():
            raise ValueError("Explicit backbone required for comparable ablations")
        digest=validate_sha(record.get("manifest_sha256"))
        signatures.add((dataset,digest,backbone))
    if len(signatures)!=1:
        raise ValueError("Ablations must share dataset, split manifest and backbone")
    return max(records,key=lambda r:r["accuracy"])


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
