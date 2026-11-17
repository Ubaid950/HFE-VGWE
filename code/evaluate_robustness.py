"""Deterministic new robustness evaluation; contains no manuscript scores."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
from evaluation_runtime import run

CONDITIONS=('clean','jpeg_q80','jpeg_q50','jpeg_q30','noise_0.01','noise_0.05','noise_0.10')


def validate_rgb(image):
    x=np.asarray(image,dtype=np.float32)
    if x.ndim!=3 or x.shape[-1]!=3 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('Explicit finite RGB[0,1] required')
    return x


def jpeg(image,quality):
    from PIL import Image
    if type(quality) is not int or quality not in {80,50,30}:raise ValueError('Supported JPEG Q80,Q50,Q30 only')
    x=validate_rgb(image);b=io.BytesIO()
    Image.fromarray(np.rint(x*255).astype(np.uint8)).save(b,format='JPEG',quality=quality,subsampling=0,optimize=False,progressive=False)
    b.seek(0)
    with Image.open(b) as im:return np.asarray(im.convert('RGB'),dtype=np.float32)/255.


def gaussian(image,sigma,seed,sample_id):
    x=validate_rgb(image)
    if sigma not in {.01,.05,.1} or type(seed) is not int or seed<0 or not isinstance(sample_id,str) or not sample_id:
        raise ValueError('Supported sigma, explicit nonnegative seed and sample ID required')
    # Same standardized noise per sample for every sigma; invariant to row order/batching.
    digest=hashlib.sha256((str(seed)+'\0'+sample_id).encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:16],'big'))
    return np.clip(x+rng.normal(0,sigma,x.shape),0,1).astype(np.float32)


def transforms(seed):
    if type(seed) is not int or seed<0:raise ValueError('Explicit nonnegative noise seed required')
    return [('clean',None)]+[(f'jpeg_q{q}',lambda x,s,q=q:jpeg(x,q)) for q in [80,50,30]]+[
        (f'noise_{s:.2f}',lambda x,sid,s=s:gaussian(x,s,seed,sid)) for s in [.01,.05,.1]]


def evaluate_robustness(config):
    if config['source_dataset_id']!=config['target_dataset_id'] or config['target_dataset_id'] not in {'ffpp','celebdf_v2'}:
        raise ValueError('Robustness uses the declared within-dataset held-out cohort')
    seed=config['noise_seed'];conditions=transforms(seed)
    out=run(config,conditions,mark_complete=False)
    try:
        rows=[]
        import csv
        from prediction_data import MODELS
        for model in [*MODELS,'EWE','VGWE']:
            clean=json.loads((out/'clean'/model/'metrics.json').read_text())['frame']
            for name,_ in conditions:
                m=json.loads((out/name/model/'metrics.json').read_text())['frame']
                rows.append({'model':model,'condition':name,'accuracy':m['accuracy'],'roc_auc':m['roc_auc'],
                    'accuracy_delta_from_clean':m['accuracy']-clean['accuracy'],
                    'roc_auc_delta_from_clean':None if m['roc_auc'] is None or clean['roc_auc'] is None else m['roc_auc']-clean['roc_auc']})
        with (out/'clean_comparison.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (out/'transformation_protocol.json').write_text(json.dumps({'conditions':CONDITIONS,'seed':seed,'input_range':[0,1],
            'order':'degrade original supplied crop before model-size bilinear resize','jpeg':'round-to-uint8, RGB,subsampling0,optimizeFalse,progressiveFalse',
            'noise':'SHA256(seed,NUL,sample_id),first128bits; NumPy default_rng; same standardized noise across sigmas; clip[0,1]',
            'classification':'new_reconstructed_degradation_protocol_not_recovered_historical_implementation'},indent=2))
        (out/'status.json').write_text(json.dumps({'status':'completed','scientific_reproduction_claim':False}))
    except Exception as e:
        (out/'status.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'error':str(e)}));raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--execute-evaluation',action='store_true')
    a=p.parse_args()
    if not a.execute_evaluation:raise ValueError('Explicit --execute-evaluation required')
    evaluate_robustness(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
