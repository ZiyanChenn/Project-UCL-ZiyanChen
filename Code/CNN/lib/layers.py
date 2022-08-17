# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow_addons as tfa


# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(0)


# import losses


# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#reflection-padding
'''
  3D Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''


class Padding3D(tf.keras.layers.Layer):
    def __init__(self, padding=(0, 0, 0), **kwargs):
        self.padding = tuple(padding)

        super(Padding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + (2 * self.padding[0]),
                input_shape[2] + (2 * self.padding[1]),
                input_shape[3] + (2 * self.padding[2]),
                input_shape[4])

    def call(self, input_tensor, *args, **kwargs):
        return tf.pad(input_tensor, [[0, 0],
                                     [self.padding[0], self.padding[0]],
                                     [self.padding[1], self.padding[1]],
                                     [self.padding[2], self.padding[2]],
                                     [0, 0]], "SYMMETRIC")


def get_padding(x, size):
    print("get_padding")

    if size[0] % 2 > 0:
        kernel_padding_1 = int(tf.math.floor(size[0] / 2.0))
    else:
        kernel_padding_1 = int(tf.math.floor(size[0] / 2.0)) - 1

    if size[1] % 2 > 0:
        kernel_padding_2 = int(tf.math.floor(size[1] / 2.0))
    else:
        kernel_padding_2 = int(tf.math.floor(size[1] / 2.0)) - 1

    if size[2] % 2 > 0:
        kernel_padding_3 = int(tf.math.floor(size[2] / 2.0))
    else:
        kernel_padding_3 = int(tf.math.floor(size[2] / 2.0)) - 1

    if (kernel_padding_1 != kernel_padding_2 or kernel_padding_1 != kernel_padding_3 or kernel_padding_2 != kernel_padding_3):
        if kernel_padding_1 != 0:
            for i in range(kernel_padding_1):
                x = Padding3D((1, 0, 0))(x)  # noqa

        if kernel_padding_2 != 0:
            for i in range(kernel_padding_2):
                x = Padding3D((0, 1, 0))(x)  # noqa

        if kernel_padding_3 != 0:
            for i in range(kernel_padding_3):
                x = Padding3D((0, 0, 1))(x)  # noqa
    else:
        if kernel_padding_1 != 0:
            for i in range(kernel_padding_1):
                x = Padding3D((1, 1, 1))(x)  # noqa

    return x


def get_dropout(x, dropout):
    print("get_dropout")

    if dropout > 0.0 and x.shape[-1] > 1:
        if len(x.shape) > 2:
            x = tf.keras.layers.SpatialDropout3D(rate=dropout)(x)
        else:
            x = tf.keras.layers.Dropout(rate=dropout)(x)

    return x


def get_convolution_layer(x, depth, size, strides, dropout):
    print("get_convolution_layer")

    x = get_padding(x, size)

    x = tf.keras.layers.Conv3D(filters=depth,
                               kernel_size=size,
                               strides=strides,
                               dilation_rate=(1, 1, 1),
                               groups=1,
                               padding="valid",
                               kernel_initializer=tf.keras.initializers.GlorotNormal(seed=999),
                               bias_initializer=tf.keras.initializers.Zeros())(x)

    x = tfa.layers.GroupNormalization(groups=1)(x)  # noqa
    x = tf.keras.layers.Lambda(tfa.activations.mish)(x)
    x = get_dropout(x, dropout)

    return x


def get_concatenate_layer(x, depth, size, strides, dropout):
    print("get_concatenate_layer")

    x = tf.keras.layers.Concatenate()(x)
    x = get_convolution_layer(x, depth, size, strides, dropout)

    return x


def get_downsample_layer(x, depth, size, strides, dropout):
    print("get_downsample_layer")

    x = get_convolution_layer(x, depth, size, strides, dropout)

    return x


def get_upsample_layer(x, depth, size, strides, dropout):
    print("get_upsample_layer")

    x = tf.keras.layers.UpSampling3D(size=strides)(x)
    x = get_convolution_layer(x, depth, size, (1, 1, 1), dropout)

    return x
