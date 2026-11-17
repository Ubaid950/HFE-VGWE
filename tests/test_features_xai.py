"""SYNTHETIC UNIT FIXTURES for HFE features, Grad-CAM and LIME."""
import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT))
from hfe_features import db2_decompose,Db2Features,SGFE,SqueezeExcitation,HFEFeatures,DB2_LOW,DB2_HIGH,FakeClassAccuracy
from training_common import build_model
from test_core import config
from ablation import FeatureConfig
from xai.gradcam import gradcam
from xai.lime_explanation import explain_lime


class FeatureTests(unittest.TestCase):
    def test_public_db2_reference_not_average_pooling(self):
        # Primary PyWavelets documented db2 symmetric fixture (not study data).
        # https://pywavelets.readthedocs.io/en/stable/regression/dwt-idwt.html
        x=np.array([3,7,1,1,-2,5,4,6],dtype=np.float64)
        low=np.array([5.65685425,7.39923721,.22414387,3.33677403,7.77817459])
        high=np.array([-2.44948974,-1.60368225,-4.44140056,-.41361256,1.22474487])
        bands=db2_decompose(tf.constant(np.outer(x,x)[None,...,None])).copy()
        for name,a,b in [("LL",low,low),("LH",low,high),("HL",high,low),("HH",high,high)]:
            np.testing.assert_allclose(bands[name].numpy()[0,:,:,0],np.outer(a,b),atol=1e-7,rtol=1e-8)
    def test_dimensions_odd_even_and_channels(self):
        for h,w in [(8,10),(7,9),(224,224)]:
            with self.subTest(shape=(h,w)):
                x=np.zeros((1,h,w,3),np.float32);x[0,h//2,w//2,1]=1
                bands=db2_decompose(x)
                for a in bands.values():
                    self.assertEqual(tuple(a.shape),(1,(h+3)//2,(w+3)//2,3))
                    np.testing.assert_array_equal(a.numpy()[...,0],0)
                    np.testing.assert_array_equal(a.numpy()[...,2],0)
    def test_training_accuracy_matches_fake_threshold_including_ties(self):
        for fake in [0,1]:
            with self.subTest(fake_index=fake):
                metric=FakeClassAccuracy(fake)
                truth=np.eye(2)[[fake,1-fake]]
                scores=np.array([[.5,.5],[.1,.9] if fake==0 else [.9,.1]])
                metric.update_state(truth,scores)
                self.assertEqual(float(metric.result()),1)
                restored=tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
                self.assertEqual(restored.fake_index,fake)
    def test_constant_db2_and_filter_moments(self):
        bands=db2_decompose(tf.ones([1,8,8,3],tf.float64))
        np.testing.assert_allclose(bands["LL"],2,atol=1e-12)
        for key in ["LH","HL","HH"]:np.testing.assert_allclose(bands[key],0,atol=1e-12)
        self.assertAlmostEqual(sum(DB2_LOW),np.sqrt(2))
        self.assertAlmostEqual(sum(DB2_HIGH),0)
        self.assertAlmostEqual(sum(k*h for k,h in enumerate(DB2_HIGH)),0)
    def test_live_pywavelets_when_available(self):
        try:import pywt
        except ImportError:self.skipTest("PyWavelets absent; public reference fixture still tested")
        x=np.random.default_rng(8).normal(size=(1,7,9,3))
        actual=db2_decompose(tf.constant(x))
        for channel in range(3):
            ll,(hl,lh,hh)=pywt.dwt2(x[0,:,:,channel],"db2",mode="symmetric")
            for name,expected in zip(["LL","LH","HL","HH"],[ll,lh,hl,hh]):
                np.testing.assert_allclose(actual[name][0,:,:,channel],expected,atol=1e-12)
    def test_rgb_then_wavelet_channel_order(self):
        x=tf.ones([1,8,8,3])*.4
        layer=HFEFeatures(dwt=True,sgfe=False,se=False)
        y=layer(x).numpy()
        self.assertEqual(y.shape,(1,8,8,6))
        np.testing.assert_array_equal(y[...,:3],x)
        np.testing.assert_allclose(y[...,3:],Db2Features()(x),atol=1e-6)
    def test_hfe_range_error_and_signed_features(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):HFEFeatures()(tf.ones([1,8,8,3])*2)
        x=tf.constant(np.random.default_rng(3).uniform(size=(1,8,8,3)),tf.float32)
        self.assertTrue(np.isfinite(HFEFeatures()(x)).all())
        with self.assertRaises(ValueError):Db2Features(coefficients=[1,1,1,1,1])
    def test_sgfe_constant_and_zero(self):
        x=tf.ones([1,8,8,3])*.5
        np.testing.assert_allclose(SGFE()(x),.85*x,atol=1e-6)
        np.testing.assert_array_equal(SGFE()(tf.zeros_like(x)),0)
    def test_sgfe_independent_channel_batch_and_gradient(self):
        x=tf.Variable(np.random.default_rng(7).uniform(size=(2,8,8,3)).astype(np.float32))
        with tf.GradientTape() as tape:
            y=SGFE()(x);loss=tf.reduce_sum(y)
        self.assertTrue(np.isfinite(tape.gradient(loss,x)).all())
        np.testing.assert_allclose(SGFE()(x[:1]),y[:1],atol=1e-7)
        with tf.GradientTape() as tape:
            z=tf.Variable(tf.zeros([1,8,8,3]));v=tf.reduce_sum(SGFE()(z))
        self.assertTrue(np.isfinite(tape.gradient(v,z)).all())
    def test_se_parameters_shape_and_serialization(self):
        layer=SqueezeExcitation(reduction=4)
        x=tf.ones([2,8,8,6]);y=layer(x)
        self.assertEqual(tuple(y.shape),tuple(x.shape))
        self.assertEqual(layer.count_params(),19) #6*1+1 +1*6+6
        self.assertEqual(len(layer.trainable_weights),4)
        self.assertTrue(np.all((y.numpy()>=0)&(y.numpy()<=1)))
        restored=tf.keras.layers.deserialize(tf.keras.layers.serialize(layer))
        self.assertEqual(restored.reduction,4)
    def test_hfe_keras_save_load_and_tf_function(self):
        inp=tf.keras.Input((8,8,3));out=HFEFeatures()(inp);model=tf.keras.Model(inp,out)
        x=tf.ones([1,8,8,3])*.5
        fn=tf.function(model)
        np.testing.assert_allclose(fn(x),model(x),atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"synthetic.keras";model.save(path)
            restored=tf.keras.models.load_model(path,safe_mode=True)
            np.testing.assert_allclose(restored(x),model(x),atol=1e-6)
    def test_all_three_corrected_backbone_interfaces(self):
        import gc
        for name in ["Xception","InceptionV3","MobileNetV1"]:
            with self.subTest(backbone=name):
                cfg=config();cfg["backbone"]=name
                model,view,backbone=build_model(cfg,FeatureConfig(True,True,True),weights=None)
                x=tf.ones([1,80,80,3])*.4
                p=model(x,training=False).numpy()
                self.assertEqual(p.shape,(1,2))
                np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
                self.assertEqual(model.get_layer("branch_projection").filters,3)
                self.assertEqual(model.get_layer("branch_projection").activation.__name__,"sigmoid")
                np.testing.assert_allclose(tf.nn.softmax(view(x,training=False)[1]),p,atol=1e-6)
                del model,view,backbone
                tf.keras.backend.clear_session();gc.collect()

    def test_model_factory_probabilities_no_download_or_fit(self):
        cfg=config()
        model,view,backbone=build_model(cfg,FeatureConfig(False,False,False),weights=None)
        x=tf.zeros([1,80,80,3]);p=model(x,training=False).numpy()
        self.assertEqual(p.shape,(1,2));np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        self.assertEqual(tuple(view(x,training=False)[1].shape),(1,2))
        self.assertEqual(model.get_layer("backbone_preprocess").scale,2)
        self.assertFalse(any(l.name=="branch_projection" for l in model.layers))
        tf.keras.backend.clear_session()


class XAITests(unittest.TestCase):
    def test_cam_shape_targets_and_forward_consistency(self):
        inputs=tf.keras.Input((8,8,3))
        conv=tf.keras.layers.Conv2D(2,1,kernel_initializer="ones",use_bias=False)(inputs)
        pooled=tf.keras.layers.GlobalAveragePooling2D()(conv)
        logits=tf.keras.layers.Dense(2,kernel_initializer=tf.keras.initializers.Constant([[1,-1],[1,-1]]))(pooled)
        view=tf.keras.Model(inputs,[conv,logits])
        images=np.ones([1,8,8,3],np.float32)
        p=tf.nn.softmax(view(images)[1]).numpy()
        result=gradcam(view,images,target_indices=[0],class_mapping={"fake":0,"real":1},
                       input_range=(0,1),expected_probabilities=p)
        self.assertEqual(result["maps"].shape,(1,8,8));self.assertTrue(result["defined"][0])
        with self.assertRaises(ValueError):
            gradcam(view,images,target_indices=[0],class_mapping={"fake":0,"real":1},
                       input_range=(0,1),expected_probabilities=[[.5,.5]])
    def test_lime_signed_format_real_library_synthetic_predictor(self):
        image=np.ones((8,8,3))*.7
        segments=np.repeat(np.repeat(np.array([[0,1],[2,3]]),4,axis=0),4,axis=1)
        def predictor(x):
            p=np.clip(x.mean((1,2,3)),0,1)
            return np.stack([1-p,p],axis=1)
        result=explain_lime(image,predictor,segments,target_index=1,
                 class_mapping={"real":0,"fake":1},input_range=(0,1),seed=7,num_samples=20)
        self.assertEqual(result["signed_map"].shape,(8,8))
        self.assertEqual(set(result["superpixel_weights"]),{0,1,2,3})
        self.assertTrue(np.isfinite(result["surrogate_fit_score"]))
        for k,v in result["superpixel_weights"].items():
            np.testing.assert_array_equal(result["signed_map"][segments==k],np.float32(v))


if __name__=="__main__":unittest.main()
