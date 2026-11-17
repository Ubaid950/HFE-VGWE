"""Corrected metrics requiring validation. Canonical labels real0/fake1.

Computes from provided predictions only, never supplies missing scores.
Undefined metrics are None with reasons, never invented AUC defaults.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from prediction_data import probabilities, read_prediction_csv


def binary_metrics(labels, p_fake, threshold=.5):
    p = probabilities(p_fake);y=np.asarray(labels)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y,[0,1]).all():
        raise ValueError("Aligned observed canonical binary labels required")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    pred=(p>=threshold).astype(int)
    tp=int(np.sum((y==1)&(pred==1)));tn=int(np.sum((y==0)&(pred==0)))
    fp=int(np.sum((y==0)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)))
    reasons={}
    def ratio(name,a,b):
        if not b:reasons[name]="Zero denominator";return None
        return a/b
    precision=ratio("precision",tp,tp+fp);recall=ratio("recall",tp,tp+fn)
    f1=ratio("f1",2*tp,2*tp+fp+fn)
    positives=int(y.sum());negatives=len(y)-positives
    auc=None
    if not positives or not negatives:reasons["roc_auc"]="Both classes required"
    else:
        # Mann–Whitney rank statistic with exact half credit for score ties.
        order=np.argsort(p,kind="stable");scores=p[order];ys=y[order]
        total=0.;negative_before=0;i=0
        while i<len(scores):
            j=i+1
            while j<len(scores) and scores[j]==scores[i]:j+=1
            pos=int(ys[i:j].sum());neg=j-i-pos
            total+=pos*(negative_before+.5*neg);negative_before+=neg;i=j
        auc=total/(positives*negatives)
    return {"n":len(y),"threshold":float(threshold),"accuracy":(tp+tn)/len(y),
            "precision":precision,"recall":recall,"f1":f1,"roc_auc":auc,
            "confusion_matrix":[[tn,fp],[fn,tp]],"class_order":["real","fake"],
            "undefined":reasons}


def aggregate_videos(predictions, *, eligible_video_ids=None):
    if predictions.labels is None or predictions.video_ids is None:
        raise ValueError("Video IDs and observed labels required")
    groups={}
    for video,y,p in zip(predictions.video_ids,predictions.labels,predictions.p_fake):
        groups.setdefault(video,[]).append((int(y),float(p)))
    rows=[]
    if eligible_video_ids is None:empty=None
    else:
        eligible=tuple(eligible_video_ids)
        if len(set(eligible))!=len(eligible) or not set(groups)<=set(eligible):
            raise ValueError("Invalid eligible-video universe")
        empty=[{"video_id":v,"reason":"No valid frame predictions"} for v in eligible if v not in groups]
    for video,values in groups.items():
        labels={y for y,p in values}
        if len(labels)!=1:raise ValueError("Inconsistent labels within video "+video)
        rows.append({"video_id":video,"label":values[0][0],
                     "n_valid_frames":len(values),
                     "p_fake":float(np.mean([p for y,p in values],dtype=np.float64)),
                     "prediction":int(np.mean([p for y,p in values],dtype=np.float64)>=.5)})
    return rows,empty


def evaluate(predictions, *, eligible_video_ids=None, exclusions=()):
    if predictions.labels is None:raise ValueError("Labels missing")
    excluded=list(exclusions)
    ids=set()
    for item in excluded:
        if not item.get("sample_id") or not item.get("reason"):raise ValueError("Exclusion needs ID and reason")
        if item["sample_id"] in predictions.sample_ids or item["sample_id"] in ids:
            raise ValueError("Exclusions overlap valid/duplicate samples")
        ids.add(item["sample_id"])
    result={"model":predictions.model_name,"dataset":predictions.dataset_id,"split":predictions.split,
            "manifest_sha256":predictions.manifest_sha256,"checkpoint_sha256":predictions.checkpoint_sha256,
            "prediction_provenance":predictions.provenance,"source_class_mapping":predictions.original_class_mapping,"frame":binary_metrics(predictions.labels,predictions.p_fake),
            "frame_exclusions":excluded,"frame_exclusion_scope":"supplied records only; missing invalid-frame universe is not inferred"}
    if predictions.video_ids is not None:
        rows,empty=aggregate_videos(predictions,eligible_video_ids=eligible_video_ids)
        result.update(video=binary_metrics([r["label"] for r in rows],[r["p_fake"] for r in rows]),
                      video_predictions=rows,empty_video_exclusions=empty,
                      video_universe_known=eligible_video_ids is not None)
    return result


def export_evaluation(predictions, output_dir, *, eligible_video_ids=None, exclusions=()):
    result=evaluate(predictions,eligible_video_ids=eligible_video_ids,exclusions=exclusions)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    rows=[{"sample_id":sid,"video_id":predictions.video_ids[i] if predictions.video_ids else "",
           "label":int(predictions.labels[i]),"p_fake":float(predictions.p_fake[i]),
           "prediction":int(predictions.p_fake[i]>=.5)}
          for i,sid in enumerate(predictions.sample_ids)]
    for name,data in [("frame_predictions.csv",rows),("video_predictions.csv",result.get("video_predictions",[]))]:
        if data:
            with (out/name).open("x",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",required=True);parser.add_argument("--output-dir",required=True)
    for name in ["model-name","dataset-id","manifest-sha256","checkpoint-sha256"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--split",choices=["train","validation","test"],required=True)
    parser.add_argument("--class-mapping",required=True,help='JSON e.g. {"real":1,"fake":0}')
    args=vars(parser.parse_args());path=args.pop("predictions");out=args.pop("output_dir")
    args["class_mapping"]=json.loads(args["class_mapping"])
    export_evaluation(read_prediction_csv(path,**args),out)


if __name__=="__main__":main()
