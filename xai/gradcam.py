"""Reconstructed explicit conv/logit Grad-CAM interface.

Caller provides a reviewed model view returning (conv_features,binary_logits).
No historical checkpoint loader: recovered Keras2.10 custom graph requires
separate compatibility validation. This module never imports TCRF.
"""
import numpy as np
import tensorflow as tf


def gradcam(cam_model, images, *, target_indices, class_mapping,
            input_range, logit_contrast=True, expected_probabilities=None):
    if set(class_mapping)!={"real","fake"} or set(class_mapping.values())!={0,1}:
        raise ValueError("Explicit real/fake model-output mapping required")
    if tuple(input_range) not in {(0,1),(0,255)}:
        raise ValueError("Explicit model input range required")
    x=tf.convert_to_tensor(images,dtype=tf.float32)
    tf.debugging.assert_rank(x,4);tf.debugging.assert_all_finite(x,"Nonfinite inputs")
    tf.debugging.assert_greater_equal(x,float(input_range[0]))
    tf.debugging.assert_less_equal(x,float(input_range[1]))
    t=tf.convert_to_tensor(target_indices,dtype=tf.int32)
    tf.debugging.assert_rank(t,1);tf.debugging.assert_equal(tf.shape(t)[0],tf.shape(x)[0])
    tf.debugging.assert_greater_equal(t,0);tf.debugging.assert_less_equal(t,1)
    with tf.GradientTape() as tape:
        # watch input as well so all-frozen reviewed models remain differentiable.
        tape.watch(x)
        conv,logits=cam_model(x,training=False)
        tf.debugging.assert_rank(conv,4)
        tf.debugging.assert_all_finite(conv,"Nonfinite CAM features")
        tf.debugging.assert_all_finite(logits,"Nonfinite CAM logits")
        tf.debugging.assert_shapes([(logits,("N",2))])
        target=tf.gather(logits,t,axis=1,batch_dims=1)
        if logit_contrast:target-=tf.gather(logits,1-t,axis=1,batch_dims=1)
        score=tf.reduce_sum(target)
    gradient=tape.gradient(score,conv)
    if gradient is None:raise ValueError("Disconnected convolution/logit model view")
    tf.debugging.assert_all_finite(gradient,"Nonfinite CAM gradient")
    weights=tf.reduce_mean(gradient,axis=[1,2],keepdims=True)
    small=tf.nn.relu(tf.reduce_sum(conv*weights,axis=-1))
    maps=tf.image.resize(small[...,None],tf.shape(x)[1:3],method="bilinear")[...,0]
    probabilities=tf.nn.softmax(logits)
    tf.debugging.assert_all_finite(probabilities,"Nonfinite CAM probabilities")
    tf.debugging.assert_all_finite(maps,"Nonfinite CAM map")
    probs=probabilities.numpy();maps=maps.numpy()
    if expected_probabilities is not None and not np.allclose(probs,expected_probabilities,atol=1e-5,rtol=0):
        raise ValueError("CAM forward view differs from reviewed predictor")
    return {"maps":maps,"probabilities":probs,"target_indices":t.numpy(),
            "defined":np.any(maps>0,axis=(1,2)),
            "target":"binary_logit_contrast" if logit_contrast else "class_logit",
            "class_mapping":dict(class_mapping),"input_range":list(input_range)}
