# Frozen evaluation contract

No checked-in configuration is approved or executable as a scientific experiment.
Paths in example files must be replaced with privately supplied actual artifacts.
Paths resolve from the invoking working directory. APIs are intentionally strict;
review declarations are external evidence, not authentication by software.

## Dataset manifest

CSV fields: sample_id,path,label; optional video_id,source_ids,sha256,split.
Paths are root-relative and unique; escaping/symlink-outside/missing paths fail.
Label numbers use target_class_mapping. If supplied, split must be test; video IDs
must be either complete and label-consistent or absent for all records. Do not
fill unknown video/source IDs. source_ids is a JSON string list of authentic,
globally dataset-scoped source groups; review any absent mappings externally.
The manifest contains actual retained files, not a sampling or split generator.
Video aggregation uses only supplied authentic IDs and explicitly lists videos
with no readable frames. Unreadable frames get exclusion reasons; hash mismatches
or frames changing between conditions fail the run. No substitute crop is made.

## Branch metadata and source validation

Exactly three models in Xception,InceptionV3,MobileNetV1 order. Each spec supplies
checkpoint,metadata,source_validation(path,sha256,provenance). Only complete
corrected .keras models are supported. Legacy HDF5/Lambda models require their
separate historical review; they are not silently adapted. The loader checks
hashes, input/output shape, corrected HFE type and embedded2*x-1 normalization,
loads without compilation in safe mode and sets trainable=false.

Metadata fields: architecture=corrected_hfe_v1,backbone,source_dataset_id,
checkpoint_sha256,class_mapping,image_size,input_contract=
rgb_float32_0_1_embedded_preprocess,selection=source_validation_only,run_id,
source_manifest_sha256. Embedded preprocessing and actual model/run must agree.
For a future completed current corrected training run, produce the sidecar with:

    PYTHONPATH=code python -c "from evaluation_runtime import write_metadata_from_run; write_metadata_from_run('PRIVATE_RUN_DIRECTORY')"

This requires the saved run/status/full-model hash and matching training-source
version; it does not train or infer. It writes once, refuses existing sidecars
and does not authorize evaluation. No historical completed run is supplied.

Validation CSVs require dataset_id equal to the declared source and split equal
to validation on every row, plus sample_id,label and p0,p1 or p_fake. Explicit
model output mappings canonicalize observed labels. Three branch datasets,
manifest/sample/video order, labels and checkpoints must agree. Both classes and
defined ACC/F1/AUC are required for VGWE. A saved file cannot be authenticated as
source-validation solely by relabeling its columns: generating-run lineage must
be reviewed externally. Fitting weights never accesses target scores or labels.

## External review artifact

Fields required by prepare(): approved_for_evaluation=true,
independence_reviewed=true,evidence_reference,source_dataset_id,target_dataset_id,
target_manifest_sha256,target_class_mapping,checkpoint_sha256s,
model_metadata_sha256s,source_validation_sha256s. Hash lists use fixed model order.
source_training_validation_group_ids optionally supplies known globally scoped
training/validation source groups for overlap checks. A scientific review must
account for any missing mappings; empty groups do not prove independence.

The reviewed target manifest, all three model metadata files, checkpoints and
source-validation artifacts are bound by hashes. Within-dataset robustness also
rejects overlap with source-validation sample/video IDs. Cross-dataset raw numeric
IDs need dataset qualification before group-overlap comparison. No string-based
preflight certifies identity, source content, rights or historical provenance.
The example review remains unapproved. The inherited scientific NO-GO is a decision,
not an authentication mechanism: changing a JSON approval flag does not resolve
the missing lineage or authorize an experiment.

## Outputs and transformation semantics

A new output directory receives provenance.json,weights inside provenance,
status.json and per-condition branch/EWE/VGWE frame CSV, optional video CSV,
metrics.json and a metrics.csv summary. Only actual evaluated samples are written.
Undefined AUC is null with a reason. Outputs include actual checkpoint/metadata,
input-manifest/content, validation/review hashes and versions; they remain private.
A failed evaluation is marked failed; no fallback score/history is generated.

Robustness uses clean; JPEG80/50/30(round RGB to uint8,subsampling0,no progressive
encoding); Gaussian sigma0.01/0.05/0.10 on[0,1], clipping after noise. Noise RNG uses
the first128bits of SHA256(seed,NUL,sample_id), is sample/order stable and reuses
standardized noise across sigma. Conditions share identical readable samples.
Degradation precedes model-size bilinear resize. Codec/library versions are
recorded; deterministic checks within the tested environment are not guarantees
of cross-version bit identity. These choices are reconstructed, not an
uncovered historical perturbation implementation. No manuscript score is stored.
