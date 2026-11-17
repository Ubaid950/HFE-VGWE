"""Synthetic regression fixtures; not research results."""
from pathlib import Path
import sys,unittest
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT)]
from ablation import select_by_validation
from xai.gradcam import gradcam
def validation_record(**changes):
    row={'split':'validation','accuracy':.8,'dataset_id':'synthetic_only',
         'manifest_sha256':'a'*64,'backbone':'Xception'}
    row.update(changes);return row
def view(bias=(0.,0.)):
    inputs=tf.keras.Input((8,8,3))
    conv=tf.keras.layers.Conv2D(2,1,kernel_initializer='ones',use_bias=False)(inputs)
    pooled=tf.keras.layers.GlobalAveragePooling2D()(conv)
    logits=tf.keras.layers.Dense(2,kernel_initializer=tf.keras.initializers.Constant([[1.,-1.],[1.,-1.]]),
                                bias_initializer=tf.keras.initializers.Constant(bias))(pooled)
    return tf.keras.Model(inputs,[conv,logits])
class AblationGuardTests(unittest.TestCase):
    def test_mixed_manifest_rejected(self):
        with self.assertRaises(ValueError):
            select_by_validation([validation_record(),validation_record(accuracy=.9,manifest_sha256='b'*64)])
    def test_mixed_backbone_rejected(self):
        with self.assertRaises(ValueError):
            select_by_validation([validation_record(),validation_record(accuracy=.9,backbone='MobileNetV1')])
    def test_mixed_dataset_rejected(self):
        with self.assertRaises(ValueError):
            select_by_validation([validation_record(),validation_record(accuracy=.9,dataset_id='other_synthetic')])
    def test_missing_comparability_evidence_rejected(self):
        for key in ['dataset_id','manifest_sha256','backbone']:
            row=validation_record();del row[key]
            with self.subTest(key=key),self.assertRaises(ValueError):
                select_by_validation([row])
    def test_valid_selection_math_unchanged(self):
        rows=[validation_record(accuracy=.8),validation_record(accuracy=.9)]
        self.assertIs(select_by_validation(rows),rows[1])
    def test_invalid_digest_rejected(self):
        with self.assertRaises(ValueError):
            select_by_validation([validation_record(manifest_sha256='not-a-sha')])
class CAMGuardTests(unittest.TestCase):
    def test_nonfinite_logits_rejected_without_optional_reference(self):
        for bad in [float('nan'),float('inf')]:
            with self.subTest(bias=bad),self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):
                gradcam(view((bad,bad)),np.ones((1,8,8,3),np.float32),target_indices=[0],
                        class_mapping={'fake':0,'real':1},input_range=(0,1))
    def test_finite_result_math_matches_analytic_reference(self):
        model=view();images=np.ones((1,8,8,3),np.float32)
        kwargs=dict(target_indices=[0],class_mapping={'fake':0,'real':1},input_range=(0,1))
        after=gradcam(model,images,**kwargs)
        # Independent analytic fixture: conv channels=3, logits=(6,-6),
        # contrast gradient=2/64 per channel, raw CAM=2*3*2/64=.1875.
        np.testing.assert_array_equal(after['maps'],np.full((1,8,8),.1875,np.float32))
        np.testing.assert_allclose(after['probabilities'],tf.nn.softmax([[6.,-6.]]).numpy(),atol=1e-7,rtol=0)
if __name__=='__main__':unittest.main()
