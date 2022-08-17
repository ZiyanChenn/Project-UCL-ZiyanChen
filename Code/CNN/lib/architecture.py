# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
# import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(0)


import parameters
import layers


def get_input(input_shape):
    print("get_input")

    x = tf.keras.layers.Input(input_shape)

    return x, x


def get_encoder(x):
    print("get_encoder")

    x = tfa.layers.GroupNormalization(groups=1, name="input")(x)  # noqa

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    unet_connections = []

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), parameters.dropout)

        unet_connections.append(x)

        x = layers.get_downsample_layer(x, layer_depth[i], (3, 3, 3), (2, 2, 2), parameters.dropout)

    return x, unet_connections


def get_latent(x):
    print("get_latent")

    layer_layers = parameters.layer_layers[-1]
    layer_depth = parameters.layer_depth[-1]

    for _ in range(layer_layers):
        x = layers.get_convolution_layer(x, layer_depth, (3, 3, 3), (1, 1, 1), parameters.dropout)

    for _ in range(layer_layers):
        x = layers.get_convolution_layer(x, layer_depth, (3, 3, 3), (1, 1, 1), parameters.dropout)

    return x


def get_decoder(x, unet_connections):
    print("get_decoder")

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    layer_layers.reverse()
    layer_depth.reverse()

    for i in range(len(layer_depth)):
        x = layers.get_upsample_layer(x, layer_depth[i], (3, 3, 3), (2, 2, 2), parameters.dropout)

        x = layers.get_concatenate_layer([x, unet_connections.pop()], layer_depth[i], (3, 3, 3), (1, 1, 1), parameters.dropout)

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), parameters.dropout)

    x = layers.get_padding(x, (3, 3, 3))
    x = tf.keras.layers.Conv3D(filters=1,
                               kernel_size=(3, 3, 3),
                               strides=(1, 1, 1),
                               dilation_rate=(1, 1, 1),
                               groups=1,
                               padding="valid",
                               kernel_initializer=tf.keras.initializers.Orthogonal(seed=999),
                               bias_initializer=tf.keras.initializers.Zeros(),
                               name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x)
    x = get_latent(x)
    output_x = get_decoder(x, unet_connections)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = tf.keras.Model(inputs=input_x, outputs=output_x)

    gc.collect()
    tf.keras.backend.clear_session()

    return model
