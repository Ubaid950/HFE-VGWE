"""Existing frame manifest contract; no dataset discovery or invented IDs."""
from pathlib import Path
import csv,json
from prediction_data import validate_mapping


def read_frames(manifest, root, mapping):
    validate_mapping(mapping);root=Path(root).resolve()
    if not root.is_dir():raise FileNotFoundError('Dataset root missing')
    with Path(manifest).open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty frame manifest')
    if not {'sample_id','path','label'}<=set(rows[0]):raise ValueError('sample_id,path,label required')
    ids=set();paths=set();video_labels={}
    has_video=[bool(r.get('video_id')) for r in rows]
    if any(has_video) and not all(has_video):raise ValueError('Partially missing video IDs')
    for r in rows:
        sid=r['sample_id'];p=Path(r['path'])
        if not sid.strip() or sid in ids:raise ValueError('Unique sample IDs required')
        if p.is_absolute():raise ValueError('Manifest paths must be relative')
        p=(root/p).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p in paths:raise ValueError('Missing/duplicate/escaping frame path')
        label=int(r['label'])
        if label not in mapping.values():raise ValueError('Invalid label')
        if 'split' in r and r['split']!='test':raise ValueError('Evaluation requires test records only')
        if all(has_video):
            v=r['video_id']
            if v in video_labels and video_labels[v]!=label:raise ValueError('Conflicting video labels')
            video_labels[v]=label
        groups=json.loads(r.get('source_ids') or '[]')
        if not isinstance(groups,list) or any(not isinstance(g,str) or not g.strip() for g in groups):
            raise ValueError('source_ids must be authentic JSON string list, or absent')
        r.update(label=label,resolved_path=str(p),source_ids=groups)
        ids.add(sid);paths.add(p)
    return rows
