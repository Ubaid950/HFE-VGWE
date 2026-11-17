"""Reconstructed shared trainer. Requires explicit config and genuine split CSV.

No training/test inference on import. No split reconstruction. Test images
are never fed to model.fit, callbacks, selection, or model evaluation here.
"""
from pathlib import Path
from dataclasses import asdict
import csv,json,hashlib,uuid,datetime,importlib.metadata,platform,traceback
import numpy as np
from prediction_data import validate_mapping,canonical_labels
from ablation import FeatureConfig

BACKBONES=("Xception","InceptionV3","MobileNetV1")
CONFIG_KEYS={"dataset_id","compression","backbone","image_size","batch_size","seed",
             "phases","label_smoothing","l2","dropouts","head_units",
             "early_stopping_patience","se_reduction","pretrained_weights",
             "identity_policy","preprocessing_attestation","augmentation","settings_provenance"}


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):h.update(data)
    return h.hexdigest()


def validate_config(config, dataset):
    if dataset not in {"celebdf_v2","ffpp"}:raise ValueError("Unsupported dataset")
    if set(config)!=CONFIG_KEYS:
        raise ValueError("Configuration fields differ: "+str(set(config)^CONFIG_KEYS))
    if config["dataset_id"]!=dataset or config["backbone"] not in BACKBONES:
        raise ValueError("Wrong dataset/backbone")
    if dataset=="ffpp" and config["compression"] not in {"c23","c40"}:
        raise ValueError("Explicit c23/c40 required; final manuscript comparison is c23")
    if dataset=="celebdf_v2" and config["compression"] is not None:
        raise ValueError("Celeb-DF compression must be null")
    for key in ["image_size","batch_size","early_stopping_patience","se_reduction"]:
        if type(config[key]) is not int or config[key]<1:raise ValueError("Invalid "+key)
    if config["pretrained_weights"] not in {None,"imagenet"}:
        raise ValueError("Only explicit ImageNet or random initialization; historical graphs cannot be converted here")
    if config["image_size"]<75:raise ValueError("At least75 pixels for all supported backbones")
    if type(config["seed"]) is not int or config["seed"]<0:raise ValueError("Explicit new-run seed required")
    if (not np.isfinite(config["label_smoothing"]) or not 0<=config["label_smoothing"]<1
        or not np.isfinite(config["l2"]) or config["l2"]<0):
        raise ValueError("Invalid regularization")
    if not config["head_units"] or any(type(v) is not int or v<1 for v in config["head_units"]):
        raise ValueError("Explicit dense-head widths required")
    if len(config["dropouts"])!=len(config["head_units"])+1 or any(
        not np.isfinite(v) or not 0<=v<1 for v in config["dropouts"]):
        raise ValueError("Dropout for pooling and each dense layer required")
    if config["identity_policy"] not in {"require_disjoint","record_overlap"}:
        raise ValueError("Explicit identity policy required; source/video grouping always disjoint")
    if not isinstance(config["preprocessing_attestation"],str) or not config["preprocessing_attestation"].strip():
        raise ValueError("Describe actual supplied face-crop preparation; not performed here")
    if not config["phases"]:raise ValueError("Explicit training phases required")
    for phase in config["phases"]:
        if set(phase)!={"epochs","optimizer","learning_rate","train_backbone","freeze_batch_norm"}:
            raise ValueError("Each phase needs epochs,optimizer,learning_rate,train_backbone,freeze_batch_norm")
        if (type(phase["epochs"]) is not int or phase["epochs"]<1
            or phase["optimizer"] not in {"Adam","RMSprop","SGD"}
            or not np.isfinite(phase["learning_rate"]) or phase["learning_rate"]<=0
            or type(phase["train_backbone"]) is not bool
            or type(phase["freeze_batch_norm"]) is not bool):
            raise ValueError("Invalid training phase")
    aug=config["augmentation"]
    if not isinstance(aug,dict) or aug.get("mode") not in {"none","keras_native"}:
        raise ValueError("Explicit augmentation mode required")
    if aug["mode"]=="none":
        if set(aug)!={"mode"}:raise ValueError("Unexpected augmentation fields")
    else:
        if set(aug)!={"mode","rotation_degrees","translation","zoom","horizontal_flip","brightness_delta"}:
            raise ValueError("Explicit native augmentation settings required")
        for key,upper in [("rotation_degrees",180),("translation",1),("zoom",1),("brightness_delta",1)]:
            if not np.isfinite(aug[key]) or not 0<=aug[key]<upper:
                raise ValueError("Invalid augmentation "+key)
        if type(aug["horizontal_flip"]) is not bool:raise ValueError("Explicit flip boolean required")
    origins=config["settings_provenance"]
    required=CONFIG_KEYS-{"settings_provenance"}
    if set(origins)!=required or any(not isinstance(v,str) or not v.strip() for v in origins.values()):
        raise ValueError("Every setting needs explicit source/manuscript/introduced provenance")
    return config


def read_split_manifest(path, root, class_mapping, *, identity_policy):
    """Existing CSV fields: sample_id,path,split,label,video_id,source_ids.

    source_ids is a JSON list of all original source-video groups involved,
    including both sources of a manipulation where applicable.
    Optional identity_ids is a JSON list; no identifiers are generated.
    """
    mapping=validate_mapping(class_mapping);root=Path(root).resolve()
    if identity_policy not in {"require_disjoint","record_overlap"}:
        raise ValueError("Unknown identity policy")
    if not root.is_dir():raise ValueError("Dataset root missing")
    with Path(path).open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    if not rows:raise ValueError("Missing/empty original split manifest; do not reconstruct")
    required={"sample_id","path","split","label","video_id","source_ids"}
    if not required<=set(rows[0]):raise ValueError("Required manifest columns missing")
    ids=set();paths=set();video_split={};source_split={};identity_splits={}
    video_label={};classes={s:set() for s in ["train","validation","test"]}
    identity_present=[bool(r.get("identity_ids")) for r in rows]
    if any(identity_present) and not all(identity_present):raise ValueError("Partially missing identity data")
    if identity_policy=="require_disjoint" and not all(identity_present):
        raise ValueError("Original identity groups unavailable; cannot verify identity disjointness")
    for r in rows:
        if not r["sample_id"] or r["sample_id"] in ids:raise ValueError("Missing/duplicate sample ID")
        ids.add(r["sample_id"])
        split=r["split"]
        if split not in classes:raise ValueError("Use explicit train/validation/test splits")
        label=int(r["label"])
        if label not in mapping.values():raise ValueError("Label outside explicit mapping")
        classes[split].add(label)
        p=(root/r["path"]).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p in paths:
            raise ValueError("Missing, duplicate or escaping dataset path")
        paths.add(p);r["resolved_path"]=str(p)
        video=r["video_id"]
        if not video:raise ValueError("Original video ID required")
        if video in video_split and video_split[video]!=split:raise ValueError("Video leakage")
        video_split[video]=split
        if video in video_label and video_label[video]!=label:raise ValueError("Inconsistent video label")
        video_label[video]=label
        groups=json.loads(r["source_ids"])
        if not isinstance(groups,list) or not groups or any(not isinstance(g,str) or not g for g in groups):
            raise ValueError("Actual source-video groups required, including multiple sources")
        for group in groups:
            if group in source_split and source_split[group]!=split:raise ValueError("Source-video leakage")
            source_split[group]=split
        if all(identity_present):
            identity=json.loads(r["identity_ids"])
            if not isinstance(identity,list) or not identity or any(not isinstance(x,str) or not x for x in identity):
                raise ValueError("Invalid identity list")
            for item in identity:identity_splits.setdefault(item,set()).add(split)
        r["label"]=label
    if any(v!={0,1} for v in classes.values()):
        raise ValueError("Every declared split must contain both real and fake; no partial manifest")
    overlaps=sorted(k for k,v in identity_splits.items() if len(v)>1)
    if identity_policy=="require_disjoint" and overlaps:raise ValueError("Identity leakage")
    return rows,{"manifest_sha256":sha256_file(path),"identity_data_available":all(identity_present),
                 "identity_policy":identity_policy,"identity_overlap_ids":overlaps,
                 "split_counts":{s:sum(r["split"]==s for r in rows) for s in classes}}


def build_model(config, feature_config, *, weights=None):
    """Returns (probability model, conv/logit CAM view, backbone).

    Branch-specific 1x1 sigmoid projection only for six-channel DWT variants.
    Its bounded range and compact SE bottleneck are NEW choices, not original.
    All three application backbones use their identical 2*x-1 preprocessing
    for x in[0,1]. No batch-dependent scaling or Lambda bytecode.
    """
    import tensorflow as tf
    from hfe_features import HFEFeatures
    inputs=tf.keras.Input((config["image_size"],config["image_size"],3),name="rgb_0_1")
    x=inputs
    aug=config["augmentation"]
    if aug["mode"]=="keras_native":
        ops=[tf.keras.layers.RandomRotation(aug["rotation_degrees"]/360,seed=config["seed"]),
             tf.keras.layers.RandomTranslation(aug["translation"],aug["translation"],seed=config["seed"]+1),
             tf.keras.layers.RandomZoom(aug["zoom"],seed=config["seed"]+2),
             tf.keras.layers.RandomBrightness(aug["brightness_delta"],value_range=(0,1),seed=config["seed"]+3)]
        if aug["horizontal_flip"]:
            ops.append(tf.keras.layers.RandomFlip("horizontal",seed=config["seed"]+4))
        x=tf.keras.Sequential(ops,name="introduced_training_augmentation")(x)
    x=HFEFeatures(**asdict(feature_config),se_reduction=config["se_reduction"],name="corrected_hfe")(x)
    if feature_config.dwt:
        x=tf.keras.layers.Conv2D(3,1,activation="sigmoid",name="branch_projection")(x)
    x=tf.keras.layers.Rescaling(2,offset=-1,name="backbone_preprocess")(x)
    factory={"Xception":tf.keras.applications.Xception,
             "InceptionV3":tf.keras.applications.InceptionV3,
             "MobileNetV1":tf.keras.applications.MobileNet}[config["backbone"]]
    backbone=factory(weights=weights,include_top=False,
                     input_shape=(config["image_size"],config["image_size"],3))
    conv=backbone(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(conv)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Dropout(config["dropouts"][0])(x)
    for units,dropout in zip(config["head_units"],config["dropouts"][1:]):
        x=tf.keras.layers.Dense(units,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(config["l2"]))(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Dropout(dropout)(x)
    logits=tf.keras.layers.Dense(2,name="class_logits")(x)
    probability=tf.keras.layers.Softmax(name="class_probabilities")(logits)
    model=tf.keras.Model(inputs,probability,name="stage3_"+config["backbone"])
    cam_view=tf.keras.Model(inputs,[conv,logits],name="stage3_cam_interface")
    return model,cam_view,backbone


def _dataset(rows, split, config, mapping):
    import tensorflow as tf
    chosen=[r for r in rows if r["split"]==split]
    # Model output indices ARE mapping indices, not canonical metric indices.
    raw_labels=[r["label"] for r in chosen]
    ds=tf.data.Dataset.from_tensor_slices(([r["resolved_path"] for r in chosen],raw_labels))
    if split=="train":ds=ds.shuffle(len(chosen),seed=config["seed"],reshuffle_each_iteration=True)
    def load(path,label):
        rgb=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        rgb.set_shape([None,None,3])
        rgb=tf.image.convert_image_dtype(rgb,tf.float32)
        rgb=tf.image.resize(rgb,[config["image_size"]]*2,method="bilinear")
        return rgb,tf.one_hot(label,2)
    ds=ds.map(load,num_parallel_calls=1)
    options=tf.data.Options();options.experimental_deterministic=True
    return ds.with_options(options).batch(config["batch_size"])


def train(config, dataset_root, manifest_path, class_mapping, output_root, feature_config):
    """Explicitly invoked only. Does not evaluate test or generate historical results."""
    import tensorflow as tf
    validate_config(config,config["dataset_id"]);mapping=validate_mapping(class_mapping)
    rows,audit=read_split_manifest(manifest_path,dataset_root,mapping,identity_policy=config["identity_policy"])
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex
    out=Path(output_root)/config["dataset_id"]/(config["compression"] or "native")/config["backbone"]/run_id
    out.mkdir(parents=True,exist_ok=False)
    metadata={"run_id":run_id,"classification":"reconstructed_corrected_architecture",
              "historical_result_reproduction":False,"config":config,"class_mapping":mapping,
              "features":asdict(feature_config),"split_audit":audit,"dataset_root":str(Path(dataset_root).resolve()),
              "python":platform.python_version(),"dependencies":{},"augmentation_semantics":"explicit config; native additive brightness differs from historical multiplicative ImageDataGenerator",
              "source_versions":{p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")}}
    for package in ["tensorflow","keras","numpy"]:
        try:metadata["dependencies"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:metadata["dependencies"][package]="unavailable"
    (out/"run_config.json").write_text(json.dumps(metadata,indent=2,allow_nan=False))
    # Preserve actual supplied IDs and split information in private run output.
    with (out/"split_manifest.csv").open("xb") as f:f.write(Path(manifest_path).read_bytes())
    def status(state,**kwargs):
        (out/"status.json").write_text(json.dumps({"status":state,**kwargs},indent=2,allow_nan=False))
    status("started")
    histories=[]
    try:
        model,_,backbone=build_model(config,feature_config,weights=config["pretrained_weights"])
        train_ds=_dataset(rows,"train",config,mapping);val_ds=_dataset(rows,"validation",config,mapping)
        class HistoryRecorder(tf.keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                record={"phase":self.phase,"epoch_in_phase":int(epoch),
                        "logs":{k:float(v) for k,v in (logs or {}).items()}}
                if not all(np.isfinite(v) for v in record["logs"].values()):
                    raise RuntimeError("Nonfinite training logs; run invalid")
                with (out/"observed_epochs.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
        best=-float("inf");best_path=None
        for index,phase in enumerate(config["phases"]):
            backbone.trainable=phase["train_backbone"]
            if phase["train_backbone"] and phase["freeze_batch_norm"]:
                for layer in backbone.layers:
                    if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
            optimizer=getattr(tf.keras.optimizers,phase["optimizer"])(learning_rate=phase["learning_rate"],clipnorm=1.)
            from hfe_features import FakeClassAccuracy
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config["label_smoothing"]),
                          metrics=[FakeClassAccuracy(mapping["fake"])])
            checkpoint=out/f"phase_{index}_validation_best.weights.h5"
            recorder=HistoryRecorder();recorder.phase=index
            history=model.fit(train_ds,validation_data=val_ds,epochs=phase["epochs"],
                callbacks=[recorder,
                           tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",mode="max",
                               patience=config["early_stopping_patience"],restore_best_weights=True),
                           tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor="val_accuracy",
                               mode="max",save_best_only=True,save_weights_only=True)])
            if not history.history.get("val_accuracy"):raise RuntimeError("No genuine validation history")
            histories.append(history.history)
            value=max(history.history["val_accuracy"])
            if value>best:best=value;best_path=checkpoint
        if best_path is None:raise RuntimeError("No valid checkpoint")
        model.load_weights(best_path)
        model.save(out/"corrected_model.keras")
        (out/"training_history.json").write_text(json.dumps(histories,indent=2,allow_nan=False))
        status("completed",selected_by="source_validation_accuracy_only",
               checkpoint=str(best_path.name),checkpoint_sha256=sha256_file(best_path),
               saved_model_sha256=sha256_file(out/"corrected_model.keras"),
               test_evaluated=False)
    except Exception as error:
        status("failed",error_type=type(error).__name__,error=str(error),
               traceback=traceback.format_exc(),test_evaluated=False)
        raise
    return out


def cli(dataset):
    import argparse
    parser=argparse.ArgumentParser(description="Reconstructed corrected trainer; no historic score guarantee")
    for name in ["config","dataset-root","train-path","validation-path","test-path",
                 "split-manifest","class-mapping","output-root"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--condition",choices=[n for n,f in __import__("ablation").CONDITIONS],default="full_hfe")
    parser.add_argument("--execute-training",action="store_true",help="Required explicit opt-in; not used during Stage3A")
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());validate_config(config,dataset)
    mapping=json.loads(args.class_mapping)
    rows,audit=read_split_manifest(args.split_manifest,args.dataset_root,mapping,identity_policy=config["identity_policy"])
    roots={s:Path(v).resolve() for s,v in [("train",args.train_path),("validation",args.validation_path),("test",args.test_path)]}
    if len(set(roots.values()))!=3 or any(
        a!=b and a.is_relative_to(b) for a in roots.values() for b in roots.values()):
        raise ValueError("Independent non-nested explicit split roots required")
    for r in rows:
        if not Path(r["resolved_path"]).is_relative_to(roots[r["split"]]):
            raise ValueError("Manifest sample outside its explicit split path")
    print(json.dumps({"validation_only_preflight":True,**audit},indent=2))
    if args.execute_training:
        flags=dict(__import__("ablation").CONDITIONS)[args.condition]
        print(train(config,args.dataset_root,args.split_manifest,mapping,args.output_root,flags))
