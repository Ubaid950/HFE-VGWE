# Explicit run configuration contract

No historical final run configuration has been authenticated. Training requires your actual split CSV, explicit paths and a complete JSON configuration. No guessed historical preset or split is provided. The entry points validate only unless --execute-training is explicitly supplied; scientific NO-GO remains in force; no training is authorized by this candidate.

Required fields (also validated by training_common.validate_config):

| Fields | Meaning / evidence boundary |
|---|---|
| dataset_id, compression, backbone | celebdf_v2 with null compression, or ffpp with explicit c23/c40; fixed Xception/InceptionV3/MobileNetV1. c23 is final local manuscript target; c40 is historical alternate, never silently pooled. |
| image_size, batch_size, seed | Explicit integers. Source Celeb image224/batch16; FF candidate image299/batch16. Final local manuscript image224/batch16. Generator seed42 is observed; global historical seed is unknown. A new seed must be marked introduced. |
| phases | Nonempty list of objects, each with epochs, optimizer (Adam/RMSprop/SGD), learning_rate, train_backbone and freeze_batch_norm. Source Celeb has30 frozen epochs +20 fine-tuning, initialLR.001, RMSprop for Xception/Inception, Adam for MobileNet; fine-tune Adam LR.0002. Local manuscript states Adam initialLR.001/max50; these are conflicting evidence targets, not silently equivalent. FF source LR.0001/EPOCHS60; a final FF run is not authenticated. Choose and document a new protocol explicitly. |
| label_smoothing, l2, dropouts, head_units | Finite values; dropouts has one value for pooling and one per head layer. Source smoothing.1, L2.0005, head4096/2048/1024/512/256 with dropout .2/.3/.3/.25/.2/.15. Local manuscript says dropout.30; full head details not established there. |
| early_stopping_patience, se_reduction | Source patience20; SE bottleneck reduction is newly introduced, not verified original. |
| pretrained_weights | Explicit null (new random initialization) or imagenet (only at future authorized training invocation). Never a historical checkpoint conversion. |
| identity_policy | require_disjoint needs authentic identity_ids for every row and rejects overlap. record_overlap records actual supplied identity overlap; source/video groups are always required and always checked. |
| preprocessing_attestation | Describe actual provided face crops. This trainer does not perform/reconstruct MTCNN video sampling, face detection, geometric alignment or original split construction. Resize of supplied RGB crops is explicitly bilinear. |
| augmentation | Explicit {"mode":"none"} OR mode keras_native with rotation_degrees, translation, zoom, horizontal_flip and brightness_delta. Native Keras augmentation is a new implementation; brightness is additive, unlike original ImageDataGenerator's multiplicative brightness_range, and shear/channel shift are not emulated. No augmentation is applied during validation/test inference. |
| settings_provenance | Dictionary covering every other configuration key. Each value must identify inspected source, manuscript statement, supplied genuine record or introduced reconstruction, and explain unresolved differences. Caller declarations do not authenticate history. |

## Split CSV

Required columns: sample_id,path,split,label,video_id,source_ids. Each row represents an actual retained frame. split is train, validation or test; labels use the explicit --class-mapping JSON. source_ids is a JSON list of ALL original source video groups involved in that sample, including multiple source identities/videos where known. Preserve these identifiers; do not synthesize them from filenames. Optional identity_ids is a JSON list. Unknown required source membership blocks preflight rather than being guessed.

All splits must contain both classes. Unique sample IDs and paths, consistent per-video labels and disjoint source/video groups are enforced. Paths must resolve inside --dataset-root and within the matching explicit --train-path/--validation-path/--test-path, whose directories cannot be nested. Existing records are used unchanged; no split generation is implemented. Identity data availability and permitted overlap are reported explicitly.

These checks validate supplied declarations, not the truth of unauthenticated manifests. Content-duplicate/face-identity detection is not implemented. Keep actual manifests and training outputs outside the public repository until privacy review.

## Usage without training

From the repository root:

    python code/train_celebdf.py --help
    python code/train_ffpp.py --help

A fully specified invocation without --execute-training performs manifest/config preflight only. No entry point executes a training run by default. When training is later authorized, unique run paths preserve the actual config, class mapping, original manifest, dependency versions, observed epochs, phase checkpoints, final model and success/failure status. Validation accuracy selects checkpoints; test data is never used by fit or selection. Evaluation is a separate supplied-prediction operation.

## Frozen evaluation interfaces

See [EVALUATION.md](EVALUATION.md) and the inactive JSON examples for frozen-source cross-dataset/robustness contracts. Source-validation CSVs and external protocol approval are required; no real evaluation config is shipped.
