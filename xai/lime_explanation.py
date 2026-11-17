"""Reconstructed signed LIME adapter based on inspected later HFE pilot.

Does not load checkpoints or select samples. Authenticated historical figures
are not claimed. Explicit reviewed predictor/range/target/segments required.
"""
import numpy as np


def explain_lime(image, predictor, segments, *, target_index, class_mapping,
                 input_range, seed, num_samples=400):
    if set(class_mapping)!={"real","fake"} or set(class_mapping.values())!={0,1}:
        raise ValueError("Explicit binary output mapping required")
    if target_index not in [0,1] or type(seed) is not int or seed<0:
        raise ValueError("Explicit target/seed required")
    if num_samples<10:raise ValueError("At least10 perturbations required; synthetic tests are not research")
    if tuple(input_range) not in {(0,1),(0,255)}:raise ValueError("Invalid explicit input range")
    rgb=np.asarray(image,dtype=np.float64);seg=np.asarray(segments)
    if (rgb.ndim!=3 or rgb.shape[-1]!=3 or not np.isfinite(rgb).all() or
        np.any(rgb<input_range[0]) or np.any(rgb>input_range[1]) or seg.shape!=rgb.shape[:2]
        or not np.issubdtype(seg.dtype,np.integer)):
        raise ValueError("Invalid RGB/segment arrays")
    ids=np.unique(seg)
    if not np.array_equal(ids,np.arange(len(ids))) or len(ids)<2:
        raise ValueError("Segments must have at least2 contiguous nonnegative IDs")
    from lime import lime_image
    def callback(images):
        values=np.asarray(predictor(np.asarray(images,dtype=np.float32)),dtype=float)
        if (values.shape!=(len(images),2) or not np.isfinite(values).all()
            or np.any((values<0)|(values>1))
            or not np.allclose(values.sum(1),1,atol=1e-6,rtol=0)):
            raise ValueError("LIME predictor must return class-aligned Nx2 probabilities")
        return values
    baseline=callback(rgb[None])[0]
    explainer=lime_image.LimeImageExplainer(kernel_width=.25,feature_selection="auto",random_state=seed)
    result=explainer.explain_instance(rgb,callback,labels=(target_index,),top_labels=None,
        hide_color=0,num_features=len(ids),num_samples=num_samples,batch_size=1,
        segmentation_fn=lambda _:seg,distance_metric="cosine")
    if not np.array_equal(result.segments,seg):raise ValueError("LIME changed fixed segmentation")
    weights={int(k):float(v) for k,v in result.local_exp[target_index]}
    if set(weights)!=set(ids):raise ValueError("Incomplete fitted superpixel coverage")
    signed=np.zeros(seg.shape,dtype=np.float32)
    for key,value in weights.items():signed[seg==key]=value
    if not np.isfinite(signed).all() or not np.isfinite(result.score):
        raise ValueError("Nonfinite signed map/surrogate fit")
    return {"signed_map":signed,"segments":seg.copy(),"superpixel_weights":weights,
            "surrogate_fit_score":float(result.score),"target_index":target_index,
            "class_mapping":dict(class_mapping),"input_range":list(input_range),
            "seed":seed,"num_samples":num_samples,"baseline_probabilities":baseline,
            "fidelity_approval":"not_established"}
