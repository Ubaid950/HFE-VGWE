"""Corrected implementation requiring validation, NOT the historical trained graph.

NHWC inputs. Db2 symmetric extension, canonical LL/LH/HL/HH (first letter
filters rows). PyWavelets dwt2 returns LL,(HL,LH,HH) under this convention.
Aggregation keeps signed coefficients. Resize/boundaries, SE bottleneck and
projection behavior are explicit reconstruction choices; historical evidence
does not establish them. No checkpoint is loaded or silently converted.
"""
import math
import tensorflow as tf

DB2_LOW = ((1-math.sqrt(3))/(4*math.sqrt(2)),
           (3-math.sqrt(3))/(4*math.sqrt(2)),
           (3+math.sqrt(3))/(4*math.sqrt(2)),
           (1+math.sqrt(3))/(4*math.sqrt(2)))
DB2_HIGH = (-DB2_LOW[3], DB2_LOW[2], -DB2_LOW[1], DB2_LOW[0])
SUBBANDS = ("LL", "LH", "HL", "HH")


def _image_tensor(inputs):
    x = tf.convert_to_tensor(inputs)
    if not x.dtype.is_floating:
        x = tf.cast(x, tf.float32)
    tf.debugging.assert_rank(x, 4, message="Expected NHWC batch")
    tf.debugging.assert_all_finite(x, "Nonfinite image/features")
    tf.debugging.assert_greater_equal(tf.shape(x)[1:3], 3)
    return x


def db2_decompose(inputs):
    """Single-level orthonormal Db2; each band has floor((N+3)/2) samples."""
    x = _image_tensor(inputs)
    dims = tf.shape(x)
    pads = tf.stack([[0, 0], [2, 2+dims[1] % 2],
                     [2, 2+dims[2] % 2], [0, 0]])
    x = tf.pad(x, pads, mode="SYMMETRIC")
    low = tf.cast(tf.constant(DB2_LOW, tf.float64), x.dtype)
    high = tf.cast(tf.constant(DB2_HIGH, tf.float64), x.dtype)
    # tf convolution is correlation: reverse the analysis filters.
    banks = [a[::-1, None] * b[None, ::-1]
             for a, b in [(low, low), (low, high), (high, low), (high, high)]]
    kernel = tf.tile(tf.stack(banks, axis=-1)[:, :, None, :],
                     [1, 1, dims[3], 1])
    y = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 2, 2, 1], padding="VALID")
    shape = tf.shape(y)
    y = tf.reshape(y, [shape[0], shape[1], shape[2], dims[3], 4])
    return {name: y[..., i] for i, name in enumerate(SUBBANDS)}


@tf.keras.utils.register_keras_serializable(package="HFEVGWEStage3")
class Db2Features(tf.keras.layers.Layer):
    def __init__(self, coefficients=(.70, .10, .10, .05, .05), **kwargs):
        super().__init__(**kwargs)
        self.coefficients = tuple(float(v) for v in coefficients)
        if (len(self.coefficients) != 5 or
            not all(math.isfinite(v) and v >= 0 for v in self.coefficients) or
            not math.isclose(sum(self.coefficients), 1, abs_tol=1e-8)):
            raise ValueError("Expected five nonnegative finite LL,LH,HL,HH,RGB coefficients summing to1")

    def call(self, inputs):
        x = _image_tensor(inputs)
        bands = db2_decompose(x)
        resized = [tf.cast(tf.image.resize(bands[k], tf.shape(x)[1:3],
                                          method="bilinear"), x.dtype)
                   for k in SUBBANDS]
        return tf.add_n([v * b for v, b in zip(self.coefficients, resized+[x])])

    def get_config(self):
        return {**super().get_config(), "coefficients": self.coefficients}


@tf.keras.utils.register_keras_serializable(package="HFEVGWEStage3")
class SGFE(tf.keras.layers.Layer):
    """Per-sample/per-channel Sobel magnitude, spatial-max normalized.

    sqrt(g²+eps)-sqrt(eps) makes constant-input response exactly zero
    and avoids undefined derivatives at zero. No signed-feature clipping.
    """
    def __init__(self, feature_weight=.85, gradient_weight=.15, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.feature_weight = float(feature_weight)
        self.gradient_weight = float(gradient_weight)
        self.epsilon = float(epsilon)
        if (not all(math.isfinite(v) and v >= 0
                    for v in [self.feature_weight, self.gradient_weight]) or
            not math.isclose(self.feature_weight+self.gradient_weight, 1, abs_tol=1e-8)
            or not math.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError("Invalid residual weights/epsilon")

    def call(self, inputs):
        x = _image_tensor(inputs)
        edges = tf.image.sobel_edges(x)
        eps = tf.cast(self.epsilon, x.dtype)
        magnitude = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1)+eps)-tf.sqrt(eps)
        normalized = tf.math.divide_no_nan(magnitude, tf.reduce_max(magnitude, [1, 2], keepdims=True))
        return self.feature_weight*x + self.gradient_weight*normalized

    def get_config(self):
        return {**super().get_config(), "feature_weight": self.feature_weight,
                "gradient_weight": self.gradient_weight, "epsilon": self.epsilon}


@tf.keras.utils.register_keras_serializable(package="HFEVGWEStage3")
class SqueezeExcitation(tf.keras.layers.Layer):
    """GAP -> learned Dense/ReLU -> Dense/sigmoid -> channel scaling."""
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = int(reduction)
        if self.reduction < 1:
            raise ValueError("reduction must be positive")

    def build(self, shape):
        channels = shape[-1]
        if channels is None:
            raise ValueError("SE requires known channel count")
        self.reduce = tf.keras.layers.Dense(max(1, channels//self.reduction), activation="relu")
        self.expand = tf.keras.layers.Dense(channels, activation="sigmoid")
        self.reduce.build((None, channels))
        self.expand.build((None, max(1, channels//self.reduction)))
        super().build(shape)

    def call(self, inputs):
        x = _image_tensor(inputs)
        gates = self.expand(self.reduce(tf.reduce_mean(x, [1, 2])))
        return x*gates[:, None, None, :]

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


@tf.keras.utils.register_keras_serializable(package="HFEVGWEStage3")
class HFEFeatures(tf.keras.layers.Layer):
    def __init__(self, dwt=True, sgfe=True, se=True, coefficients=(.70,.10,.10,.05,.05),
                 feature_weight=.85, gradient_weight=.15, se_reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.flags = dict(dwt=bool(dwt), sgfe=bool(sgfe), se=bool(se))
        self.coefficients = tuple(coefficients)
        self.feature_weight, self.gradient_weight = feature_weight, gradient_weight
        self.se_reduction = se_reduction
        self.wavelet = Db2Features(coefficients)
        self.gradient = SGFE(feature_weight, gradient_weight)
        self.recalibrate = SqueezeExcitation(se_reduction)

    def build(self, shape):
        shape = tf.TensorShape(shape)
        if shape.rank != 4 or shape[-1] != 3:
            raise ValueError("HFE requires explicit NHWC RGB shape")
        if self.flags["dwt"]:
            self.wavelet.build(shape)
        channels = 6 if self.flags["dwt"] else 3
        transformed_shape = shape[:-1].concatenate([channels])
        if self.flags["sgfe"]:
            self.gradient.build(transformed_shape)
        if self.flags["se"]:
            self.recalibrate.build(transformed_shape)
        super().build(shape)

    def call(self, inputs):
        x = _image_tensor(inputs)
        tf.debugging.assert_equal(tf.shape(x)[-1], 3, message="RGB input required")
        tf.debugging.assert_greater_equal(x, tf.cast(0, x.dtype))
        tf.debugging.assert_less_equal(x, tf.cast(1, x.dtype))
        if self.flags["dwt"]:
            x = tf.concat([x, self.wavelet(x)], axis=-1)  # RGB then RGB wavelet
        if self.flags["sgfe"]:
            x = self.gradient(x)
        if self.flags["se"]:
            x = self.recalibrate(x)
        return x

    def get_config(self):
        return {**super().get_config(), **self.flags, "coefficients": self.coefficients,
                "feature_weight": self.feature_weight, "gradient_weight": self.gradient_weight,
                "se_reduction": self.se_reduction}


@tf.keras.utils.register_keras_serializable(package="HFEVGWEStage3")
class FakeClassAccuracy(tf.keras.metrics.Mean):
    """Training selection metric with the same fake>=.50 rule as evaluation."""
    def __init__(self, fake_index, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fake_index = int(fake_index)
        if self.fake_index not in (0, 1):
            raise ValueError("Explicit fake output index0 or1 required")

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = y_true[..., self.fake_index] >= .5
        decision = y_pred[..., self.fake_index] >= .5
        return super().update_state(tf.cast(tf.equal(truth, decision), self.dtype),
                                    sample_weight=sample_weight)

    def get_config(self):
        return {**super().get_config(), "fake_index": self.fake_index}
