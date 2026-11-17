"""Reconstructed corrected-model evaluation boundary, not a historical loader.

No execution on import. Review declarations are required external inputs, not
proof supplied by software. New evaluations remain scientifically NO-GO until
lineage/protocol review. Output directories must be new, private run locations.
"""
from pathlib import Path
from dataclasses import asdict
import csv,json
import numpy as np
from prediction_data import MODELS,PredictionSet,fake_column,validate_mapping,aligned
from fusion_ewe_vgwe import fit_validation_weights,equal_weight
from evaluate_intradataset import export_evaluation
from training_common import sha256_file
from datasets.manifest import read_frames
from datasets.wilddeepfake import read_wilddeepfake


def checked_file(path,digest):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    if sha256_file(p)!=digest:raise ValueError('Artifact hash mismatch: '+p.name)
    return p


def load_bundle(spec,source_dataset):
    metadata_path=Path(spec['metadata']);m=json.loads(metadata_path.read_text())
    required={'architecture','backbone','source_dataset_id','checkpoint_sha256','class_mapping',
              'image_size','input_contract','selection','run_id','source_manifest_sha256'}
    if not required<=set(m):raise ValueError('Incomplete model metadata')
    if (m['architecture']!='corrected_hfe_v1' or m['source_dataset_id']!=source_dataset
        or m['backbone'] not in MODELS or m['selection']!='source_validation_only'
        or m['input_contract']!='rgb_float32_0_1_embedded_preprocess' or not m['run_id']):
        raise ValueError('Unsupported/unreviewed model contract; historical graphs need separate reviewed adapters')
    validate_mapping(m['class_mapping'])
    if type(m['image_size']) is not int or m['image_size']<75:raise ValueError('Invalid input size')
    from prediction_data import validate_sha
    validate_sha(m['source_manifest_sha256']);validate_sha(m['checkpoint_sha256'])
    checkpoint=checked_file(spec['checkpoint'],m['checkpoint_sha256'])
    if checkpoint.suffix!='.keras':raise ValueError('Reviewed full corrected .keras model required')
    return {'metadata':m,'checkpoint':checkpoint,'metadata_sha256':sha256_file(metadata_path),'metadata_path':metadata_path}


def load_predictor(bundle):
    import tensorflow as tf
    import hfe_features  # register corrected serializable layers; never execute legacy Lambda
    model=tf.keras.models.load_model(bundle['checkpoint'],compile=False,safe_mode=True)
    m=bundle['metadata'];n=m['image_size']
    if tuple(model.input_shape)!=(None,n,n,3) or tuple(model.output_shape)!=(None,2):
        raise ValueError('Model signature disagrees with metadata')
    pre=model.get_layer('backbone_preprocess').get_config()
    if pre.get('scale')!=2 or pre.get('offset')!=-1:raise ValueError('Missing corrected embedded normalization')
    if not isinstance(model.get_layer('corrected_hfe'),hfe_features.HFEFeatures):
        raise ValueError('Corrected HFE layer type mismatch')
    name=m['backbone']
    candidates=[layer for layer in model.layers if (layer.name=={'Xception':'xception','InceptionV3':'inception_v3'}.get(name) or (name=='MobileNetV1' and layer.name.startswith('mobilenet_1.00_')))]
    if len(candidates)!=1:raise ValueError('Backbone graph disagrees with model metadata')
    model.trainable=False
    return model


def prepare(config):
    source=config['source_dataset_id'];target=config['target_dataset_id']
    if source not in {'ffpp','celebdf_v2'}:raise ValueError('Unsupported source dataset')
    mapping=validate_mapping(config['target_class_mapping'])
    manifest=Path(config['target_manifest']);review_path=Path(config['protocol_review'])
    review=json.loads(review_path.read_text());digest=sha256_file(manifest)
    if (review.get('approved_for_evaluation') is not True or not review.get('evidence_reference')
        or review.get('source_dataset_id')!=source or review.get('target_dataset_id')!=target
        or review.get('target_manifest_sha256')!=digest
        or review.get('independence_reviewed') is not True):
        raise ValueError('External source/target lineage and protocol review missing/mismatched')
    specs=config['models']
    if len(specs)!=3:raise ValueError('Exactly three frozen branches required')
    bundles=[load_bundle(s,source) for s in specs]
    if tuple(b['metadata']['backbone'] for b in bundles)!=MODELS:raise ValueError('Fixed branch order required')
    hashes=[b['metadata']['checkpoint_sha256'] for b in bundles]
    if len(set(hashes))!=3 or review.get('checkpoint_sha256s')!=hashes:raise ValueError('Checkpoint review mismatch')
    if review.get('model_metadata_sha256s')!=[b['metadata_sha256'] for b in bundles] or review.get('target_class_mapping')!=mapping:
        raise ValueError('Model metadata/class mapping review mismatch')
    from prediction_data import read_prediction_csv
    validation=[];validation_hashes=[]
    for b,spec in zip(bundles,specs):
        m=b['metadata'];v=spec['source_validation'];path=checked_file(v['path'],v['sha256'])
        if not v.get('provenance'):raise ValueError('Authentic source-validation provenance required')
        with path.open(newline='') as f:declared=list(csv.DictReader(f))
        if not declared or any(r.get('dataset_id')!=source or r.get('split')!='validation' for r in declared):
            raise ValueError('Source-validation CSV must explicitly declare dataset_id and validation split')
        validation_hashes.append(v['sha256'])
        validation.append(read_prediction_csv(path,model_name=m['backbone'],dataset_id=source,split='validation',
          manifest_sha256=m['source_manifest_sha256'],checkpoint_sha256=m['checkpoint_sha256'],
          class_mapping=m['class_mapping'],provenance=v['provenance']))
    if review.get('source_validation_sha256s')!=validation_hashes:raise ValueError('Validation review mismatch')
    fitted=fit_validation_weights(validation) # No target labels used here.
    reader=read_wilddeepfake if target=='wilddeepfake' else read_frames
    rows=reader(manifest,config['dataset_root'],mapping)
    if any(set(r['source_ids'])&set(review.get('source_training_validation_group_ids',[])) for r in rows):
        raise ValueError('Target source groups overlap source training/validation')
    if source==target:
        if set(r['sample_id'] for r in rows)&set(fitted.validation_sample_ids):raise ValueError('Validation/test sample leakage')
        if fitted.validation_video_ids and any(r.get('video_id') in fitted.validation_video_ids for r in rows):
            raise ValueError('Validation/test video leakage')
    return rows,bundles,fitted,{'target_manifest_sha256':digest,'protocol_review_sha256':sha256_file(review_path),
        'source_validation_sha256s':validation_hashes,'source_dataset_id':source,'target_dataset_id':target,
        'classification':'new_corrected_evaluation_not_historical_reproduction',
        'models':[{'checkpoint_sha256':b['metadata']['checkpoint_sha256'],'metadata_sha256':b['metadata_sha256'],
                   'backbone':b['metadata']['backbone'],'run_id':b['metadata']['run_id'],
                   'model_class_mapping':b['metadata']['class_mapping'],'image_size':b['metadata']['image_size'],
                   'input_contract':b['metadata']['input_contract']} for b in bundles]}


def read_rgb(path, expected_sha=None):
    from PIL import Image
    import io,hashlib
    raw=Path(path).read_bytes()
    if expected_sha is not None and hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise RuntimeError('Input frame content changed or manifest hash mismatch')
    with Image.open(io.BytesIO(raw)) as im:
        im.load();return np.asarray(im.convert('RGB'),dtype=np.float32)/255.


def valid_frames(rows):
    valid=[];excluded=[]
    for r in rows:
        try:
            r['input_sha256']=r.get('sha256') or sha256_file(r['resolved_path'])
            read_rgb(r['resolved_path'],r['input_sha256'])
        except (OSError,ValueError) as e:excluded.append({'sample_id':r['sample_id'],'reason':type(e).__name__+': '+str(e)})
        else:valid.append(r)
    if not valid:raise ValueError('No readable frames; no replacements')
    return valid,excluded


def infer(rows,bundle,predictor,mapping,manifest_sha,dataset,transform=None):
    # Bounded-memory single-frame inference; model preprocessing is embedded.
    import tensorflow as tf
    scores=[];m=bundle['metadata']
    for r in rows:
        x=read_rgb(r['resolved_path'],r['input_sha256'])
        if transform is not None:x=transform(x,r['sample_id'])
        x=tf.image.resize(x,(m['image_size'],m['image_size']),method='bilinear').numpy()[None]
        p=np.asarray(predictor(x,training=False))
        scores.append(float(fake_column(p,m['class_mapping'])[0]))
    return PredictionSet(m['backbone'],dataset,'test',manifest_sha,m['checkpoint_sha256'],mapping,
         tuple(r['sample_id'] for r in rows),scores,[r['label'] for r in rows],
         tuple(r['video_id'] for r in rows) if rows[0].get('video_id') else None,'actual_frozen_corrected_checkpoint_inference')


def cross_weighted(predictions,fitted):
    items=aligned(predictions,require_labels=True)
    if tuple(i.model_name for i in items)!=MODELS or tuple(i.checkpoint_sha256 for i in items)!=fitted.checkpoint_sha256s:
        raise ValueError('Frozen source checkpoint identities changed')
    return np.tensordot(np.asarray(fitted.weights),np.stack([i.p_fake for i in items]),axes=1)


def export_condition(items,fitted,out,excluded,eligible_videos):
    first=items[0];weights={'EWE':equal_weight(items),'VGWE':cross_weighted(items,fitted)}
    for item in items:export_evaluation(item,out/item.model_name,exclusions=excluded,eligible_video_ids=eligible_videos)
    combined_hash=__import__('hashlib').sha256(''.join(fitted.checkpoint_sha256s).encode()).hexdigest()
    for name,prob in weights.items():
        pred=PredictionSet(name,first.dataset_id,'test',first.manifest_sha256,combined_hash,{'real':0,'fake':1},
                           first.sample_ids,prob,first.labels,first.video_ids,'actual_three_branch_fusion')
        export_evaluation(pred,out/name,exclusions=excluded,eligible_video_ids=eligible_videos)
    # One metrics CSV per condition; values computed only from inference above.
    with (out/'metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','level','n','accuracy','precision','recall','f1','roc_auc']);w.writeheader()
        for name in [*MODELS,'EWE','VGWE']:
            m=json.loads((out/name/'metrics.json').read_text())
            for level in ['frame','video']:
                if level in m:w.writerow({'model':name,'level':level,**{k:m[level][k] for k in ['n','accuracy','precision','recall','f1','roc_auc']}})


def run(config,conditions, *, mark_complete=True):
    rows,bundles,fitted,provenance=prepare(config)
    out=Path(config['output_dir'])
    if out.exists():raise FileExistsError(out)
    valid,excluded=valid_frames(rows)
    models=[load_predictor(b) for b in bundles]
    out.mkdir(parents=True,exist_ok=False)
    (out/'status.json').write_text(json.dumps({'status':'started'}))
    try:
        provenance['config']=config;provenance['fusion']=asdict(fitted);provenance['ewe_weights']=[1/3]*3;provenance['exclusions']=excluded;provenance['input_content_sha256s']={r['sample_id']:r['input_sha256'] for r in valid}
        import PIL,tensorflow as tf
        provenance['versions']={'numpy':np.__version__,'Pillow':PIL.__version__,'tensorflow':tf.__version__}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
        videos=tuple(dict.fromkeys(r['video_id'] for r in rows)) if rows[0].get('video_id') else None
        for name,transform in conditions:
            dest=out/name;dest.mkdir()
            items=[infer(valid,b,m,config['target_class_mapping'],provenance['target_manifest_sha256'],
                         config['target_dataset_id'],transform) for b,m in zip(bundles,models)]
            export_condition(items,fitted,dest,excluded,videos)
        (out/'status.json').write_text(json.dumps({'status':'completed' if mark_complete else 'inference_completed','scientific_reproduction_claim':False}))
    except Exception as e:
        (out/'status.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'error':str(e)}));raise
    return out


def write_metadata_from_run(run_dir):
    """Opt-in sidecar exporter for a completed current corrected training run."""
    root=Path(run_dir);m=json.loads((root/'run_config.json').read_text());s=json.loads((root/'status.json').read_text())
    if (s['status']!='completed' or s['selected_by']!='source_validation_accuracy_only'
        or m['classification']!='reconstructed_corrected_architecture'
        or m['source_versions'].get('training_common.py')!=sha256_file(Path(__file__).with_name('training_common.py'))):
        raise ValueError('Completed current corrected run evidence required')
    checked_file(root/'corrected_model.keras',s['saved_model_sha256'])
    result={'architecture':'corrected_hfe_v1','backbone':m['config']['backbone'],'source_dataset_id':m['config']['dataset_id'],
        'checkpoint_sha256':s['saved_model_sha256'],'class_mapping':m['class_mapping'],'image_size':m['config']['image_size'],
        'input_contract':'rgb_float32_0_1_embedded_preprocess','selection':'source_validation_only','run_id':m['run_id'],
        'source_manifest_sha256':m['split_audit']['manifest_sha256']}
    path=root/'evaluation_model_metadata.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    return path
