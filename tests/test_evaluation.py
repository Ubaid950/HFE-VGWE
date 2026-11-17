"""SYNTHETIC SOFTWARE FIXTURES ONLY: no manuscript results or original faces."""
from pathlib import Path
import sys,tempfile,csv,json,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code'),str(ROOT)]
from datasets.wilddeepfake import read_wilddeepfake
from evaluation_runtime import read_rgb,valid_frames,cross_weighted,prepare,load_bundle,write_metadata_from_run
from evaluate_cross_dataset import validate_direction,DIRECTIONS
from evaluate_robustness import jpeg,gaussian,transforms,CONDITIONS
from prediction_data import PredictionSet,MODELS
from fusion_ewe_vgwe import fit_validation_weights

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def manifest(self,entries):
        p=self.root/'frames.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
        return p
    def frames(self):
        from PIL import Image
        Image.fromarray(np.full((9,11,3),127,np.uint8)).save(self.root/'synthetic.png')
        return [{'sample_id':'synthetic/a','path':'synthetic.png','label':1}]
    def test_manifest_only_wilddeepfake(self):
        r=read_wilddeepfake(self.manifest(self.frames()),self.root,{'real':1,'fake':0})
        self.assertEqual(r[0]['label'],1);self.assertNotIn('video_id',r[0])
        self.assertEqual(read_rgb(r[0]['resolved_path']).shape,(9,11,3))
    def test_missing_manifest_and_root(self):
        with self.assertRaises(FileNotFoundError):read_wilddeepfake(self.root/'missing',self.root,{'real':0,'fake':1})
        with self.assertRaises(FileNotFoundError):read_wilddeepfake(self.root/'missing',self.root/'absent',{'real':0,'fake':1})
    def test_path_escape_and_duplicate(self):
        row=self.frames()[0];row['path']='../outside.png'
        with self.assertRaises(ValueError):read_wilddeepfake(self.manifest([row]),self.root,{'real':0,'fake':1})
        row=self.frames()[0]
        with self.assertRaises(ValueError):read_wilddeepfake(self.manifest([row,row]),self.root,{'real':0,'fake':1})
    def test_partial_video_and_invalid_class(self):
        row=self.frames()[0];row['label']=2
        with self.assertRaises(ValueError):read_wilddeepfake(self.manifest([row]),self.root,{'real':0,'fake':1})
        from PIL import Image
        Image.fromarray(np.zeros((9,11,3),np.uint8)).save(self.root/'second.png')
        pair=[{'sample_id':'a','path':'synthetic.png','label':0,'video_id':'known'}, {'sample_id':'b','path':'second.png','label':1,'video_id':''}]
        with self.assertRaises(ValueError):read_wilddeepfake(self.manifest(pair),self.root,{'real':0,'fake':1})
    def test_unreadable_exclusion_no_replacement(self):
        entries=self.frames();(self.root/'bad.png').write_bytes(b'not an image')
        entries.append({'sample_id':'synthetic/b','path':'bad.png','label':0})
        rows=read_wilddeepfake(self.manifest(entries),self.root,{'real':0,'fake':1});valid,excluded=valid_frames(rows)
        self.assertEqual(len(valid),1);self.assertEqual(excluded[0]['sample_id'],'synthetic/b')
    def test_input_hash_change_is_fatal(self):
        from training_common import sha256_file
        row=self.frames()[0];digest=sha256_file(self.root/'synthetic.png')
        (self.root/'synthetic.png').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):read_rgb(self.root/'synthetic.png',digest)
    def test_four_directions_only(self):
        for a,b in DIRECTIONS:validate_direction({'source_dataset_id':a,'target_dataset_id':b})
        with self.assertRaises(ValueError):validate_direction({'source_dataset_id':'wilddeepfake','target_dataset_id':'ffpp'})
    def test_review_missing_blocks_evaluation(self):
        p=self.manifest(self.frames());(self.root/'review.json').write_text('{}')
        with self.assertRaises(ValueError):prepare({'source_dataset_id':'ffpp','target_dataset_id':'wilddeepfake','target_class_mapping':{'real':0,'fake':1},'target_manifest':str(p),'protocol_review':str(self.root/'review.json')})
    def test_missing_metadata_and_checkpoint_error(self):
        with self.assertRaises(FileNotFoundError):load_bundle({'metadata':self.root/'absent'},'ffpp')
        (self.root/'metadata').write_text('{}')
        with self.assertRaises(ValueError):load_bundle({'metadata':self.root/'metadata'},'ffpp')
    def sets(self,dataset,split,labels):
        return [PredictionSet(n,dataset,split,('a' if dataset=='ffpp' else 'b')*64,str(i+1)*64,{'real':0,'fake':1},('synthetic/0','synthetic/1'),[.1,.9],labels,None,'synthetic_test_only') for i,n in enumerate(MODELS)]
    def test_cross_weights_frozen_and_labels_irrelevant(self):
        fitted=fit_validation_weights(self.sets('ffpp','validation',[0,1]))
        a=cross_weighted(self.sets('celebdf_v2','test',[0,1]),fitted)
        b=cross_weighted(self.sets('celebdf_v2','test',[1,0]),fitted)
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a,[.1,.9])
    def test_cross_alignment_failure(self):
        fitted=fit_validation_weights(self.sets('ffpp','validation',[0,1]));p=self.sets('celebdf_v2','test',[0,1])
        p[1]=PredictionSet(MODELS[1],'celebdf_v2','test','b'*64,'2'*64,{'real':0,'fake':1},('wrong','order'),[.1,.9],[0,1])
        with self.assertRaises(ValueError):cross_weighted(p,fitted)
    def test_jpeg_determinism_shape_and_range(self):
        x=np.linspace(0,1,9*11*3,dtype=np.float32).reshape(9,11,3)
        for q in [80,50,30]:
            a=jpeg(x,q);np.testing.assert_array_equal(a,jpeg(x,q));self.assertEqual(a.shape,x.shape);self.assertTrue(((a>=0)&(a<=1)).all())
        with self.assertRaises(ValueError):jpeg(x,90)
    def test_noise_seed_sample_pairing_and_range(self):
        x=np.full((16,16,3),.5,np.float32)
        a=gaussian(x,.01,17,'synthetic/a');np.testing.assert_array_equal(a,gaussian(x,.01,17,'synthetic/a'))
        self.assertFalse(np.array_equal(a,gaussian(x,.01,17,'synthetic/b')))
        np.testing.assert_allclose((a-.5)*5,gaussian(x,.05,17,'synthetic/a')-.5,atol=2e-7)
        self.assertTrue(((gaussian(x,.1,1,'a')>=0)&(gaussian(x,.1,1,'a')<=1)).all())
    def test_degradation_conditions_and_bad_range(self):
        self.assertEqual(tuple(n for n,_ in transforms(1)),CONDITIONS)
        with self.assertRaises(ValueError):gaussian(np.ones((2,2,3))*255,.01,1,'a')
        with self.assertRaises(ValueError):transforms(-1)
    def test_corrected_xception_target_view(self):
        import tensorflow as tf
        from training_common import build_model
        from test_core import config
        from ablation import FeatureConfig
        from xai.model_interface import xception_cam_view
        from xai.gradcam import gradcam
        c=config();c["backbone"]="Xception";model,_,_=build_model(c,FeatureConfig(True,True,True),weights=None)
        x=np.zeros((1,c['image_size'],c['image_size'],3),np.float32)
        for target in ['block14_sepconv2','block14_sepconv2_act']:
            view=xception_cam_view(model,target_layer=target)
            result=gradcam(view,x,target_indices=[0],class_mapping={'real':1,'fake':0},input_range=(0,1),expected_probabilities=model(x,training=False).numpy())
            self.assertTrue(np.isfinite(result['maps']).all())
        with self.assertRaises(ValueError):xception_cam_view(model,target_layer='guessed')
        from evaluation_runtime import load_predictor
        path=self.root/'synthetic_corrected.keras';model.save(path)
        loaded=load_predictor({'checkpoint':path,'metadata':{'image_size':c['image_size'],'backbone':'Xception'}})
        np.testing.assert_allclose(loaded(x,training=False).numpy(),model(x,training=False).numpy(),atol=1e-6,rtol=0)
        with self.assertRaises(ValueError):load_predictor({'checkpoint':path,'metadata':{'image_size':c['image_size'],'backbone':'MobileNetV1'}})
        tf.keras.backend.clear_session()

    def evaluation_config(self,target='celebdf_v2'):
        from PIL import Image
        from training_common import sha256_file
        for i in [0,1]:Image.fromarray(np.full((8,8,3),i*255,np.uint8)).save(self.root/f'synthetic_{i}.png')
        manifest=self.manifest([{'sample_id':f'{target}/synthetic/{i}','path':f'synthetic_{i}.png','label':i,'video_id':f'{target}/video/{i}','source_ids':json.dumps([f'{target}/source/{i}'])} for i in [0,1]])
        specs=[]
        for i,name in enumerate(MODELS):
            checkpoint=self.root/f'{name}.keras';checkpoint.write_bytes(('synthetic_mock_checkpoint_'+str(i)).encode())
            metadata={'architecture':'corrected_hfe_v1','backbone':name,'source_dataset_id':'ffpp','checkpoint_sha256':sha256_file(checkpoint),'class_mapping':{'real':0,'fake':1},'image_size':80,'input_contract':'rgb_float32_0_1_embedded_preprocess','selection':'source_validation_only','run_id':'synthetic_fixture','source_manifest_sha256':'a'*64}
            mp=self.root/f'{name}.json';mp.write_text(json.dumps(metadata))
            vp=self.root/f'{name}.csv'
            with vp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','label','p_fake','dataset_id','split','video_id']);w.writeheader()
                for j in [0,1]:w.writerow({'sample_id':f'ffpp/validation/{j}','label':j,'p_fake':.1+.8*j,'dataset_id':'ffpp','split':'validation','video_id':f'ffpp/validation_video/{j}'})
            specs.append({'metadata':str(mp),'checkpoint':str(checkpoint),'source_validation':{'path':str(vp),'sha256':sha256_file(vp),'provenance':'synthetic_software_fixture_only'}})
        review={'approved_for_evaluation':True,'independence_reviewed':True,'evidence_reference':'SYNTHETIC_TEST_ONLY','source_dataset_id':'ffpp','target_dataset_id':target,'target_manifest_sha256':sha256_file(manifest),'target_class_mapping':{'real':0,'fake':1},'checkpoint_sha256s':[sha256_file(s['checkpoint']) for s in specs],'model_metadata_sha256s':[sha256_file(s['metadata']) for s in specs],'source_validation_sha256s':[s['source_validation']['sha256'] for s in specs],'source_training_validation_group_ids':['ffpp/source/train']}
        rp=self.root/'review.json';rp.write_text(json.dumps(review))
        return {'source_dataset_id':'ffpp','target_dataset_id':target,'dataset_root':str(self.root),'target_manifest':str(manifest),'target_class_mapping':{'real':0,'fake':1},'protocol_review':str(rp),'models':specs,'output_dir':str(self.root/'synthetic_mock_output'),'noise_seed':17}
    @staticmethod
    def mock_predictor(x,training=False):
        p=float(np.asarray(x).mean());return np.array([[1-p,p]])
    def test_mocked_cross_pipeline_exports_aligned_outputs(self):
        from evaluate_cross_dataset import evaluate_cross_dataset
        c=self.evaluation_config()
        with patch('evaluation_runtime.load_predictor',return_value=self.mock_predictor):out=evaluate_cross_dataset(c)
        self.assertEqual(json.loads((out/'status.json').read_text())['status'],'completed')
        m=json.loads((out/'clean/VGWE/metrics.json').read_text());self.assertEqual(m['frame']['n'],2);self.assertEqual(m['video']['n'],2)
        with (out/'clean/VGWE/frame_predictions.csv').open() as f:actual=list(csv.DictReader(f))
        self.assertEqual([r['sample_id'] for r in actual],['celebdf_v2/synthetic/0','celebdf_v2/synthetic/1'])
        with self.assertRaises(FileExistsError):evaluate_cross_dataset(c)
    def test_mocked_robustness_fixed_cohort_and_provenance(self):
        from evaluate_robustness import evaluate_robustness
        c=self.evaluation_config('ffpp')
        with patch('evaluation_runtime.load_predictor',return_value=self.mock_predictor):out=evaluate_robustness(c)
        ids=[]
        for condition in CONDITIONS:
            with (out/condition/'VGWE/frame_predictions.csv').open() as f:ids.append([r['sample_id'] for r in csv.DictReader(f)])
        self.assertTrue(all(x==ids[0] for x in ids));self.assertTrue((out/'clean_comparison.csv').exists());self.assertTrue((out/'transformation_protocol.json').exists())
    def test_metadata_tamper_and_test_derived_weights_rejected(self):
        from training_common import sha256_file
        c=self.evaluation_config();mp=Path(c['models'][0]['metadata']);m=json.loads(mp.read_text());m['class_mapping']={'real':1,'fake':0};mp.write_text(json.dumps(m))
        with self.assertRaises(ValueError):prepare(c)
        c=self.evaluation_config();vp=Path(c['models'][0]['source_validation']['path']);vp.write_text(vp.read_text().replace(',validation,',',test,'))
        c['models'][0]['source_validation']['sha256']=sha256_file(vp)
        with self.assertRaises(ValueError):prepare(c)
    def test_missing_checkpoint_and_group_overlap_rejected(self):
        c=self.evaluation_config();Path(c['models'][0]['checkpoint']).unlink()
        with self.assertRaises(FileNotFoundError):prepare(c)
        c=self.evaluation_config();p=Path(c['protocol_review']);m=json.loads(p.read_text());m['source_training_validation_group_ids']=['celebdf_v2/source/0'];p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):prepare(c)

    def test_run_failure_status_never_fallback_metrics(self):
        from evaluate_cross_dataset import evaluate_cross_dataset
        c=self.evaluation_config()
        def broken(x,training=False):raise RuntimeError('synthetic simulated inference failure')
        with patch('evaluation_runtime.load_predictor',return_value=broken),self.assertRaises(RuntimeError):evaluate_cross_dataset(c)
        out=Path(c['output_dir']);self.assertEqual(json.loads((out/'status.json').read_text())['status'],'failed')
        self.assertFalse((out/'clean/VGWE/metrics.json').exists())
    def test_metadata_export_rejects_uncompleted_run(self):
        (self.root/'run_config.json').write_text('{}');(self.root/'status.json').write_text('{"status":"failed"}')
        with self.assertRaises(ValueError):write_metadata_from_run(self.root)

    def test_metadata_export_positive_shape_and_no_overwrite(self):
        from evaluation_runtime import write_metadata_from_run
        from training_common import sha256_file
        checkpoint=self.root/'corrected_model.keras';checkpoint.write_bytes(b'synthetic metadata-export fixture; not a loadable model')
        meta={'classification':'reconstructed_corrected_architecture','source_versions':{'training_common.py':sha256_file(ROOT/'code/training_common.py')},'config':{'backbone':'Xception','dataset_id':'ffpp','image_size':224},'class_mapping':{'real':1,'fake':0},'run_id':'synthetic_fixture_only','split_audit':{'manifest_sha256':'a'*64}}
        status={'status':'completed','selected_by':'source_validation_accuracy_only','saved_model_sha256':sha256_file(checkpoint)}
        (self.root/'run_config.json').write_text(json.dumps(meta));(self.root/'status.json').write_text(json.dumps(status))
        p=write_metadata_from_run(self.root);actual=json.loads(p.read_text())
        self.assertEqual(actual['checkpoint_sha256'],status['saved_model_sha256']);self.assertEqual(actual['class_mapping'],meta['class_mapping'])
        with self.assertRaises(FileExistsError):write_metadata_from_run(self.root)

    def test_source_validation_target_mapping_are_separate(self):
        import numpy as np
        from test_evaluation import EvaluationTests
        from evaluation_runtime import prepare,infer,valid_frames
        # Reuse a clearly synthetic configuration, never an actual trained model.
        fixture=EvaluationTests();fixture.setUp()
        try:
            c=fixture.evaluation_config();rows,bundles,fitted,prov=prepare(c);valid,_=valid_frames(rows)
            b=dict(bundles[0]);b['metadata']=dict(b['metadata'],class_mapping={'real':1,'fake':0})
            p=infer(valid,b,lambda x,training=False:np.array([[.8,.2]]),{'real':0,'fake':1},'b'*64,'celebdf_v2')
            np.testing.assert_allclose(p.p_fake,[.8,.8]);np.testing.assert_array_equal(p.labels,[0,1])
        finally:fixture.tearDown()

if __name__=='__main__':unittest.main()


