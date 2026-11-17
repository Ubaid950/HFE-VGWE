# HFE-Xception explanation interfaces

No TCRF classifier, face parser or restricted example is included. No scientific
explanation is generated automatically. These are reconstructed Grad-CAM and
LIME interfaces, not authenticated figure reproduction.

For an already reviewed corrected Xception model, use
model_interface.xception_cam_view(model,target_layer=...) with an explicit
block14_sepconv2(pre-activation) or block14_sepconv2_act(post-activation) target.
The pre-activation target follows later saved historical references. The view
shares weights and is tested against the model's forward probabilities. Grad-CAM
uses the requested class logit or target-minus-other logit contrast; input range,
class indices and target IDs are explicit. Finite logits/gradients/maps are
required; expected_probabilities can verify predictor parity.

Historical HFE-Xception input[0,255] and corrected input[0,1] must not be mixed.
No historical HDF5/Lambda loader is shipped or silently replaced. Explanation of
one constituent does not explain VGWE.

LIME fits the actual superpixel surrogate using supplied fixed contiguous
segments, class target, predictor range and seed. Defaults:400perturbations,
black hiding, cosine distance, kernel0.25. Signed coefficients, baseline
probabilities and actual surrogate fit are returned; fit is not a faithfulness
approval. The exact original400-perturbation figure generator is unverified.
