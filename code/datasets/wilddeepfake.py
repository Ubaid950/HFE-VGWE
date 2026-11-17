"""WildDeepfake adapter: original local layout is unverified; manifest required.

CSV sample_id,path,label, optional video_id/source_ids. Paths are root-relative;
labels use explicit metadata mapping. No folder guessing, redistribution or
reconstruction of unavailable video identities. Decode failures are recorded by
shared evaluation loader without replacement.
"""
from datasets.manifest import read_frames


def read_wilddeepfake(manifest, root, class_mapping):
    return read_frames(manifest, root, class_mapping)
