# Provenance

All included numerical algorithms/trainers/evaluators are corrected or
reconstructed research software. None is claimed to be the authenticated final
original manuscript implementation. Standard imported TensorFlow/Keras, NumPy,
Pillow, LIME and related packages remain third-party dependencies; their code is
not vendored here. Ownership/license approval remains with the author.

Native Db2, six-channel concatenation, learned SE and branch projection agree
with the intended paper architecture at the formula level. Inspected historical
checkpoints instead use three-channel statistical attention, with wavelet off
or a pooling approximation. The preserved historical Inception loader's saved
224→299resize operation is separate from this corrected .keras interface; it is
not shipped here or silently substituted. Historical tensor-load checks do not
prove original-runtime prediction equivalence.

The two original ablation/ensemble summary CSVs are inconsistent with the
manuscript and correspond to a legacy notebook output context, including
MobileNetV1 ablation and test-selected/test-weighted ensemble behavior. They are
not redistributed as paper evidence. No missing raw predictions or results have
been reconstructed to reconcile the mismatch.

Internal manuscript extraction, detailed traceability, original file hashes,
CSV discrepancies and immutable copy mappings are retained outside this
candidate. The original working repository and all historical evidence remain
unchanged. Public code serialization identifiers are preserved intentionally;
cosmetic removal of historical namespace tags could break compatibility.

A newly trained corrected model, new source grouping, new extraction procedure
or updated explanation protocol requires separately identified experiments and
new validation. Software fixtures and agreement with formulas do not validate
old performance numbers. No public GitHub release, repository URL, DOI or license
is asserted by this local candidate.
