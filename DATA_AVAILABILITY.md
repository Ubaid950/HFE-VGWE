# Data access and sharing

Obtain the intended dataset versions from their providers under applicable terms:

- [FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [WildDeepfake](https://github.com/xingjunm/wild-deepfake)

No raw video/crop, restricted face example, identity manifest, checkpoint,
experimental prediction array or unreconciled result CSV is included. Dataset
provider links are not substitutes for the study's original code/results.

Supply configurable private roots and authentic manifests. Never infer original
source/person IDs or timestamps from visually similar crops. WildDeepfake local
organization remains unverified; its adapter intentionally requires explicit
sample/path/label metadata and accepts video identifiers only if available.
Unreadable frames receive reasons, not replacements. Model output encoding and
target-label encoding are explicit and may differ.

Original study-specific split/extraction/run and full prediction evidence remain
incomplete. No complete manuscript-supporting deposit or publicly available code
URL is claimed. Future shareable outputs require privacy, rights and provenance
review; keep generated scores and identifying metadata outside this folder.
No author license, copyright permission, dataset sharing right or journal
approval is invented here. See PROVENANCE.md and LIMITATIONS.md.
