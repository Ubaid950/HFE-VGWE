# HFE-VGWE research software

A small, private review candidate for corrected/reconstructed HFE-VGWE code.
**This is not an authenticated reproduction of the manuscript experiments.**
No dataset, checkpoint, experiment CSV, prediction array or restricted face image
is included. No replacement split is active. Scientific training/evaluation and
actual public publication remain unapproved.

## Supported methods

| Area | Implementation / boundary |
|---|---|
| HFE | Per-channel native Db2 LL/LH/HL/HH; signed resized D3 aggregation (0.70/0.10/0.10/0.05/0.05); RGB+DWT six channels; Sobel SGFE (0.85/0.15); trainable SE |
| CNNs | Corrected Xception, InceptionV3, MobileNetV1 with independent learned 1×1 sigmoid channel projections for six-channel variants and embedded input normalization |
| Training / ablation | Explicit config/manifests and separate run directories; six feature conditions; source-validation selection; no automatic dataset split |
| Fusion | EWE exactly1/3 each; VGWE normalized mean of source-validation ACC/Fake F1/Fake AUC; fixed named members and aligned samples |
| Evaluation | Genuine supplied-prediction metrics, fixed0.50 threshold, valid-frame mean video scores; four frozen-source cross-dataset directions; manifest-driven WildDeepfake |
| Robustness | Clean, JPEGQ80/Q50/Q30, Gaussian sigma0.01/0.05/0.10; paired sample-stable noise and input-content protection |
| XAI | Reviewed Xception conv/logit views, Grad-CAM and actual LIME superpixel fits |

These implementations agree with important manuscript formulas but do not prove
that the historical reported studies used them. Original checkpoints inspected
have different features from corrected HFE. See [provenance](PROVENANCE.md),
[limitations](LIMITATIONS.md), [dataset access](DATA_AVAILABILITY.md),
[training configuration](configs/README.md),
[frozen evaluation](configs/EVALUATION.md) and [XAI](xai/README.md).

## Installation and software tests

Use a compatible existing Python environment. The candidate was tested with
Python3.12.3, TensorFlow2.21.0/Keras3.15.1, NumPy2.5.3 and Pillow12.3.0.
requirements.txt describes the full test/XAI dependency set; it is unpinned and
is not a historical lockfile. No package installation or clean-install test was
performed. PyWavelets is optional; its live comparison is skipped if unavailable.

From this directory, using python from the compatible environment:

    CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
    python code/train_celebdf.py --help
    python code/train_ffpp.py --help
    python code/evaluate_intradataset.py --help
    python code/evaluate_cross_dataset.py --help
    python code/evaluate_robustness.py --help

Latest review: 79 tests ran; 78 passed and the optional live PyWavelets check
was skipped because the dependency is unavailable. No failures or errors.
This is a self-contained software check, not experimental reproduction.

These lightweight tests/help interfaces are verified. Tests use synthetic frames,
random-weight models and mock inference in temporary directories; their outputs
are not research evidence. No arrays or facial pixels are bundled.
No scientific inference or training was executed to prepare this candidate.

## Using the interfaces after separate scientific approval

Training requires a complete explicit configuration, actual root-relative split
manifest, real/fake output mapping, source/video groups and all split paths.
Without --execute-training the trainer performs preflight. The explicit execute
flag is not scientific approval. No historical training preset is supplied.

For a future completed corrected run, evaluation_runtime.write_metadata_from_run
can emit a checked sidecar. Evaluation requires three frozen corrected .keras
models in Xception/InceptionV3/MobileNetV1 order, their verified metadata,
authentic source-validation CSVs and a reviewed target-manifest/config bundle.
The JSON examples are incomplete and approval=false. Do not turn a missing source
mapping into a guessed ID or consider a changed approval flag proof of lineage.
Real completed-run commands/evaluations are untested and scientifically NO-GO.

Cross-dataset directions are FF++→Celeb-DF, FF++→WildDeepfake,
Celeb-DF→FF++ and Celeb-DF→WildDeepfake, without target adaptation. Report frame
metrics for the manuscript's cross-dataset comparison. The generic API can also
emit video summaries when authentic IDs are supplied; those extra summaries are
not reported manuscript cross-dataset results.

All per-sample outputs and provenance belong in a new private caller-selected
directory outside this candidate, not in a shared results folder. No measured
paper scores are hardcoded. Undefined metrics, unreadable frames, missing
artifacts and failed execution are recorded or rejected rather than replaced.

## Six-Region Semantic Attribution — Code Availability

The six-region semantic attribution analysis is described in the accompanying manuscript. Its original implementation is not distributed in this GitHub repository. Researchers wishing to reproduce or extend this analysis should obtain the implementation directly from the original code author, subject to the author's permission and applicable licensing conditions.

## Scope limitations

Original16-frame/MTCNN/largest-face preprocessing history is missing; this
candidate consumes already prepared crops and does not implement that complete
video-extraction stage. Table2's additional ResNet50/VGG16/VGG19/MobileNetV2
benchmarks and Table7's complete sensitivity-study runner are not implemented.
Conflicting original ablation/ensemble summary CSVs are excluded. Their source
pipeline is not equivalent to the manuscript's stated Db2/SE and fixed ensemble.
No full Table1–8 result authentication or Figures8–9 reproduction is claimed.

Training/evaluation remain blocked by the inherited Celeb/FF++lineage/protocol
restrictions. No license or dataset redistribution permission is invented. This
folder is for local owner review only; it has not been published or uploaded.
No LICENSE file is supplied: the owner must confirm code ownership and choose
an authorized software license before public release. Dataset access terms do
not grant permission to redistribute datasets or private research artifacts.
