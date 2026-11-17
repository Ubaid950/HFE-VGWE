"""Explicit Xception target view, sharing an already reviewed model's weights.

No historical checkpoint deserializer or automatic target-layer policy. The
pre-activation target is evidenced by later historical references; corrected
factory's final activation is a different, explicitly named choice.
"""


def xception_cam_view(model, *, target_layer):
    import tensorflow as tf
    if target_layer not in {'block14_sepconv2','block14_sepconv2_act'}:
        raise ValueError('Explicit reviewed pre/post activation target required')
    model.get_layer('corrected_hfe')
    backbone=model.get_layer('xception')
    logits_layer=model.get_layer('class_logits')
    target=backbone.get_layer(target_layer)
    prefix=tf.keras.Model(model.inputs,model.get_layer('backbone_preprocess').output)
    extractor=tf.keras.Model(backbone.inputs,[target.output,backbone.output])
    rgb=tf.keras.Input(model.input_shape[1:]);features=prefix(rgb,training=False)
    conv,x=extractor(features,training=False)
    seen=False
    for layer in model.layers:
        if layer is backbone:seen=True;continue
        if not seen:continue
        x=layer(x,training=False)
        if layer is logits_layer:break
    if not seen:raise ValueError('Backbone interface missing')
    return tf.keras.Model(rgb,[conv,x],name='reviewed_'+target_layer+'_cam')
