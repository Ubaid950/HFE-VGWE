"""SYNTHETIC UNIT FIXTURES ONLY. No manuscript metric reproduction."""
import os,sys,unittest,tempfile,json,csv,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT))
from prediction_data import PredictionSet,MODELS,fake_column,canonical_labels,read_prediction_csv
from evaluate_intradataset import binary_metrics,evaluate,aggregate_videos,export_evaluation
from fusion_ewe_vgwe import equal_weight,fit_validation_weights,weighted_predict,load_and_verify_weights
from training_common import validate_config,read_split_manifest,train,CONFIG_KEYS
from ablation import FeatureConfig,CONDITIONS,ablation_plan,select_by_validation

SHA="a"*64
def predictions(split="validation", values=None, ids=None, labels=(0,0,1,1),mapping=None):
    values=values or [[.1,.2,.8,.9],[.2,.7,.6,.8],[.1,.3,.4,.9]]
    return [PredictionSet(model,"synthetic_dataset",split,SHA,str(i+1)*64,
             mapping or {"real":0,"fake":1},
             tuple(ids or [split+str(j) for j in range(4)]),p,labels,
             tuple(split+"v"+str(j) for j in range(4)),"synthetic_unit_fixture")
            for i,(model,p) in enumerate(zip(MODELS,values))]

def config(dataset="celebdf_v2"):
    cfg=dict(dataset_id=dataset,compression="c23" if dataset=="ffpp" else None,
         backbone="MobileNetV1",image_size=80,batch_size=2,seed=7,
         phases=[dict(epochs=1,optimizer="Adam",learning_rate=.001,
                      train_backbone=False,freeze_batch_norm=True)],
         label_smoothing=.1,l2=.0005,dropouts=[.0,.0],head_units=[4],
         early_stopping_patience=1,se_reduction=4,pretrained_weights=None,
         identity_policy="require_disjoint",augmentation={"mode":"none"},preprocessing_attestation="synthetic fixture, not research")
    cfg["settings_provenance"]={k:"introduced SYNTHETIC UNIT FIXTURE" for k in cfg}
    return cfg

def manifest_fixture(tmp):
    root=Path(tmp);rows=[]
    for s in ["train","validation","test"]:
        for cls in [0,1]:
            rel=f"{s}/{cls}.png";p=root/rel;p.parent.mkdir(exist_ok=True);p.write_bytes(b"synthetic placeholder; never decoded")
            rows.append(dict(sample_id=f"{s}{cls}",path=rel,split=s,label=cls,
                 video_id=f"{s}v{cls}",source_ids=json.dumps([f"{s}source{cls}"]),
                 identity_ids=json.dumps([f"{s}identity{cls}"])))
    path=root/"manifest.csv"
    def save():
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save();return root,path,rows,save

class MetricsTests(unittest.TestCase):
    def test_known_confusion_and_auc(self):
        m=binary_metrics([0,0,1,1],[.1,.8,.4,.9])
        self.assertEqual(m["confusion_matrix"],[[1,1],[1,1]])
        self.assertEqual(m["accuracy"],.5);self.assertEqual(m["precision"],.5)
        self.assertEqual(m["recall"],.5);self.assertEqual(m["f1"],.5);self.assertEqual(m["roc_auc"],.75)
    def test_auc_ties(self):
        self.assertEqual(binary_metrics([0,1],[.5,.5])["roc_auc"],.5)
    def test_undefined_auc_not_fallback(self):
        m=binary_metrics([0,0],[.1,.2]);self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["precision"]);self.assertIsNone(m["recall"]);self.assertIsNone(m["f1"])
        self.assertIn("roc_auc",m["undefined"])
    def test_threshold_boundary(self):
        self.assertEqual(binary_metrics([0,1],[.49,.5])["accuracy"],1)
    def test_invalid_scores_and_labels(self):
        for y,p in [([0,1],[.1,np.nan]),([0,2],[.1,.9]),([0],[1.1]),([],[]),([0,None],[.1,.9])]:
            with self.subTest(y=y),self.assertRaises(ValueError):binary_metrics(y,p)
    def test_inverted_mapping(self):
        np.testing.assert_array_equal(canonical_labels([0,1],{"fake":0,"real":1}),[1,0])
        np.testing.assert_array_equal(fake_column([[.8,.2]],{"fake":0,"real":1}),[.8])
        with self.assertRaises(ValueError):fake_column([[.8,.4]],{"fake":0,"real":1})
    def test_video_means(self):
        p=PredictionSet("m","d","test",SHA,"b"*64,{"real":0,"fake":1},
                        ("a","b","c"),[.1,.7,.9],[0,0,1],("v","v","w"))
        rows,empty=aggregate_videos(p,eligible_video_ids=["v","w","empty"])
        self.assertAlmostEqual(rows[0]["p_fake"],.4);self.assertEqual(rows[0]["n_valid_frames"],2)
        self.assertEqual(empty,[{"video_id":"empty","reason":"No valid frame predictions"}])
        self.assertEqual(evaluate(p)["video"]["accuracy"],1)
    def test_video_inconsistent_label_rejected(self):
        p=PredictionSet("m","d","test",SHA,"b"*64,{"real":0,"fake":1},("a","b"),[.1,.9],[0,1],("v","v"))
        with self.assertRaises(ValueError):aggregate_videos(p)
    def test_missing_labels(self):
        p=PredictionSet("m","d","test",SHA,"b"*64,{"real":0,"fake":1},("a",),[.5])
        with self.assertRaises(ValueError):evaluate(p)
    def test_exclusion_overlap(self):
        p=predictions()[0]
        with self.assertRaises(ValueError):evaluate(p,exclusions=[{"sample_id":p.sample_ids[0],"reason":"invalid"}])
    def test_export_exact_probabilities_and_no_overwrite(self):
        p=predictions()[0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"out";export_evaluation(p,path)
            with (path/"frame_predictions.csv").open() as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["p_fake"]),p.p_fake[0])
            with self.assertRaises(FileExistsError):export_evaluation(p,path)
    def test_csv_explicit_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"p.csv";p.write_text("sample_id,label,p0,p1\na,0,0.8,0.2\nb,1,0.1,0.9\n")
            got=read_prediction_csv(p,model_name="m",dataset_id="d",split="test",
                    manifest_sha256=SHA,checkpoint_sha256="b"*64,class_mapping={"real":1,"fake":0})
            np.testing.assert_array_equal(got.labels,[1,0]);np.testing.assert_allclose(got.p_fake,[.8,.1])

class FusionTests(unittest.TestCase):
    def test_missing_and_zero_validation_scores(self):
        with self.assertRaises(ValueError):fit_validation_weights(None)
        with self.assertRaises(ValueError):fit_validation_weights([])
        reverse=[[.9,.8,.2,.1]]*3
        with self.assertRaises(ValueError):fit_validation_weights(predictions(values=reverse))
    def test_persisted_weights_verified_against_original_validation(self):
        p=predictions();w=fit_validation_weights(p)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"weights.json";w.save(path)
            self.assertEqual(load_and_verify_weights(path,p).weights,w.weights)
            record=json.loads(path.read_text());record["weights"]=[1/3]*3
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):load_and_verify_weights(path,p)

    def test_equal_mean(self):
        p=predictions()
        np.testing.assert_allclose(equal_weight(p),np.mean([i.p_fake for i in p],axis=0))
    def test_exact_formula(self):
        p=predictions();w=fit_validation_weights(p)
        scores=np.array([(m["accuracy"]+m["f1"]+m["roc_auc"])/3 for m in w.validation_metrics])
        np.testing.assert_allclose(w.weights,scores/scores.sum())
        q=predictions("test")
        np.testing.assert_allclose(weighted_predict(q,w),np.array(w.weights)@np.array([i.p_fake for i in q]))
    def test_test_labels_cannot_fit(self):
        with self.assertRaises(ValueError):fit_validation_weights(predictions("test"))
        with self.assertRaises(ValueError):fit_validation_weights(predictions("train"))
    def test_order_mismatch_rejected(self):
        p=predictions();q=list(p)
        q[1]=PredictionSet(MODELS[1],"synthetic_dataset","validation",SHA,"2"*64,
                 {"real":0,"fake":1},tuple(reversed(p[0].sample_ids)),p[1].p_fake,p[1].labels,p[1].video_ids)
        with self.assertRaises(ValueError):equal_weight(q)
    def test_model_selection_forbidden(self):
        with self.assertRaises(ValueError):equal_weight(list(reversed(predictions())))
    def test_overlap_and_checkpoint_changes(self):
        p=predictions();w=fit_validation_weights(p)
        with self.assertRaises(ValueError):weighted_predict(predictions("test",ids=p[0].sample_ids),w)
        q=predictions("test")
        object.__setattr__(q[0],"checkpoint_sha256","f"*64)
        with self.assertRaises(ValueError):weighted_predict(q,w)
    def test_undefined_validation_auc(self):
        with self.assertRaises(ValueError):fit_validation_weights(predictions(labels=[0,0,0,0]))
    def test_no_validation_labels(self):
        with self.assertRaises(ValueError):fit_validation_weights(predictions(labels=None))
    def test_validation_test_video_leakage_rejected(self):
        p=predictions();w=fit_validation_weights(p);q=predictions("test")
        for item in q:object.__setattr__(item,"video_ids",p[0].video_ids)
        with self.assertRaisesRegex(ValueError,"video IDs overlap"):weighted_predict(q,w)
    def test_inference_does_not_use_test_labels(self):
        w=fit_validation_weights(predictions())
        np.testing.assert_array_equal(weighted_predict(predictions("test",labels=[0,0,1,1]),w),
                                      weighted_predict(predictions("test",labels=[1,1,0,0]),w))
    def test_cross_branch_mapping(self):
        p=predictions();q=predictions(mapping={"real":1,"fake":0},labels=[1,1,0,0])
        np.testing.assert_array_equal(fit_validation_weights(p).weights,fit_validation_weights(q).weights)

class TrainingContractTests(unittest.TestCase):
    def test_explicit_settings_required(self):
        validate_config(config(),"celebdf_v2")
        cfg=config();del cfg["seed"]
        with self.assertRaises(ValueError):validate_config(cfg,"celebdf_v2")
    def test_ffpp_compression(self):
        cfg=config("ffpp");validate_config(cfg,"ffpp");cfg["compression"]="unknown"
        with self.assertRaises(ValueError):validate_config(cfg,"ffpp")
    def test_split_manifest_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,rows,save=manifest_fixture(tmp);before=path.read_bytes()
            loaded,audit=read_split_manifest(path,root,{"real":0,"fake":1},identity_policy="require_disjoint")
            self.assertEqual(len(loaded),6);self.assertEqual(path.read_bytes(),before)
            self.assertEqual(audit["split_counts"],dict(train=2,validation=2,test=2))
    def test_source_video_identity_leakage(self):
        for column in ["video_id","source_ids","identity_ids"]:
            with self.subTest(column=column),tempfile.TemporaryDirectory() as tmp:
                root,path,rows,save=manifest_fixture(tmp);rows[2][column]=rows[0][column];save()
                with self.assertRaises(ValueError):read_split_manifest(path,root,{"real":0,"fake":1},identity_policy="require_disjoint")
    def test_missing_original_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,rows,save=manifest_fixture(tmp);rows[0]["source_ids"]="[]";save()
            with self.assertRaises(ValueError):read_split_manifest(path,root,{"real":0,"fake":1},identity_policy="require_disjoint")
    def test_ablation_isolation(self):
        plan=ablation_plan(SHA)
        self.assertEqual(len(plan),6);self.assertEqual(len({r["checkpoint_namespace"] for r in plan}),6)
        self.assertEqual(len({r["manifest_sha256"] for r in plan}),1)
        self.assertTrue(all(r["backbone"]=="Xception" and r["status"]=="planned_not_executed" for r in plan))
        self.assertEqual(CONDITIONS[-1][1],FeatureConfig(True,True,True))
        with self.assertRaises(ValueError):select_by_validation([{"split":"test","accuracy":1}])
    def test_fit_failure_and_no_test_dataset_access(self):
        from unittest.mock import Mock
        model=Mock()
        model.fit.side_effect=RuntimeError("synthetic fit failure")
        backbone=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root,path,rows,save=manifest_fixture(tmp)
            with patch("training_common.build_model",return_value=(model,None,backbone)), patch("training_common._dataset",return_value="SYNTHETIC_TRAIN_OR_VALIDATION") as ds:
                with self.assertRaisesRegex(RuntimeError,"synthetic fit failure"):
                    train(config(),root,path,{"real":0,"fake":1},root/"runs",CONDITIONS[0][1])
                self.assertEqual([call.args[1] for call in ds.call_args_list],["train","validation"])
            status=json.loads(next((root/"runs").rglob("status.json")).read_text())
            self.assertEqual(status["status"],"failed")
            self.assertFalse(list((root/"runs").rglob("training_history.json")))

    def test_failure_is_recorded_and_raised_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,rows,save=manifest_fixture(tmp)
            with patch("training_common.build_model",side_effect=RuntimeError("synthetic controlled failure")):
                with self.assertRaisesRegex(RuntimeError,"synthetic controlled failure"):
                    train(config(),root,path,{"real":0,"fake":1},root/"runs",CONDITIONS[0][1])
            statuses=list((root/"runs").rglob("status.json"));self.assertEqual(len(statuses),1)
            status=json.loads(statuses[0].read_text());self.assertEqual(status["status"],"failed")
            self.assertFalse(list((root/"runs").rglob("training_history.json")))
            self.assertFalse(list((root/"runs").rglob("*.keras")))

if __name__=="__main__":unittest.main()
