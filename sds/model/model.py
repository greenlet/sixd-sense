import collections
import dataclasses as dcls
import functools
import math
import os
import string
from typing import Tuple, List, Dict, Any, Optional, Callable

from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sds.model.params import ScaledParams
from sds.utils.utils import compose


MOMENTUM = 0.997
EPSILON = 1e-4


def prefixer(prefix: str = ''):
    if not prefix:
        return lambda name: name
    return lambda name: f'{prefix}/{name}'


def bifpn_init(feature_maps: List[Any], num_channels: int, name: str) -> List[Any]:
    pname = prefixer(name)
    res = []
    for i, features_in in enumerate(feature_maps):
        features_out = tf.keras.layers.SeparableConv2D(
            num_channels, kernel_size=3, strides=1, padding='same', name=pname(f'SeparableConv2D{i}'))(features_in)
        res.append(features_out)
    return res


def SeparableConvBlock(num_channels: int, kernel_size: int, strides: int, gn_enabled: bool, name: str):
    pname = prefixer(name)
    f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
        use_bias=True, name=pname('SeparableConv2D'))
    if not gn_enabled:
        return f1
    
    f2 = tfa.layers.GroupNormalization(groups=num_channels // 16)
    return compose(f1, f2)


class BifpnWeightedAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.w = None

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name='w',
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        # Input is a list of similar shapes: [TensorShape[m, n], TensorShape[m, n], ...]
        return input_shape[0]

    def get_config(self):
        config = {
            **super().get_config(),
            'epsilon': self.epsilon,
        }
        return config


def bifpn_merge(feature_maps_cur_level: List[Any], feature_map_other_level, upsample_other: bool, num_channels: int,
        gn_enabled: bool, name: str):
    pname = prefixer(name)
    if upsample_other:
        feature_map_resampled = tf.keras.layers.UpSampling2D(name=pname('UpSampling2D'))(feature_map_other_level)
    else:
        feature_map_resampled = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same', name=pname('MaxPooling2D'))(feature_map_other_level)
    
    feature_map = BifpnWeightedAdd(name=pname('BifpnWeightedAdd'))(feature_maps_cur_level + [feature_map_resampled])
    feature_map = tf.keras.activations.swish(feature_map)
    feature_map = SeparableConvBlock(
        num_channels=num_channels, kernel_size=3, strides=1, gn_enabled=gn_enabled, name=pname('SeparableConvBlock'))(feature_map)
    return feature_map


def bifpn_top_down(feature_maps: List[Any], num_channels: int, gn_enabled: bool, name: str) -> List[Any]:
    pname = prefixer(name)
    features_out = [feature_maps[0]]
    for i in range(1, len(feature_maps)):
        features_merged = bifpn_merge(
            feature_maps_cur_level=[feature_maps[i]],
            feature_map_other_level=features_out[-1],
            upsample_other=True,
            num_channels=num_channels,
            gn_enabled=gn_enabled,
            name=pname(f'merge{i}'))
        features_out.append(features_merged)
    return features_out


def bifpn_bottom_up(feature_maps: List[List[Any]], num_channels: int, gn_enabled: bool, name: str) -> List[Any]:
    pname = prefixer(name)
    features_out = [feature_maps[0][0]]
    for i in range(1, len(feature_maps)):
        features_merged = bifpn_merge(
            feature_maps_cur_level=feature_maps[i],
            feature_map_other_level=features_out[-1],
            upsample_other=False,
            num_channels=num_channels,
            gn_enabled=gn_enabled,
            name=pname(f'merge{i}'),
        )
        features_out.append(features_merged)
    return features_out


def bifpn_layer(feature_maps: List[Any], num_channels: int, gn_enabled: bool = True, name: str = 'BifpnLayer') -> List[Any]:
    pname = prefixer(name)
    feature_maps = list(reversed(feature_maps))
    features_top_down = bifpn_top_down(feature_maps, num_channels, gn_enabled=gn_enabled, name=pname('BifpnTopDown'))
    features_mid = []
    n_maps = len(feature_maps)
    for i in range(n_maps):
        if 0 < i < n_maps - 1:
            features_mid.append([feature_maps[i], features_top_down[i]])
        else:
            features_mid.append([features_top_down[i]])
    
    features_mid = list(reversed(features_mid))
    features_bottom_up = bifpn_bottom_up(features_mid, num_channels, gn_enabled=gn_enabled, name=pname('BifpnBottomUp'))
    
    return features_bottom_up


def final_upscale(fpn_feature_maps: List[Any], num_channels: int, name: str):
    pname = prefixer(name)
    feature_maps_in = []
    for i, feature_map in enumerate(reversed(fpn_feature_maps)):
        feature_map_in = tf.keras.layers.SeparableConv2D(
            num_channels, kernel_size=3, strides=1, padding='same', name=pname(f'SeparableConv2D{i}'))(feature_map)
        feature_maps_in.append(feature_map_in)
    
    feature_maps = bifpn_top_down(feature_maps_in, num_channels=num_channels, gn_enabled=False, name=pname('BifpnTopDown'))
    feature_map_out = feature_maps[-1]
    feature_map_out = tf.keras.layers.Conv2DTranspose(num_channels, kernel_size=3, strides=2, padding='same', name=pname('Upscale'))(feature_map_out)
    return feature_map_out


if __name__ == '__main__':
    phi = 0
    num_classes = 20
    freeze_bn = True
    params = ScaledParams(phi)
    image_input = tf.keras.Input(params.input_shape)
    _, bb_feature_maps = params.backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)

    fpn_init_feature_maps = bifpn_init(bb_feature_maps, params.bifpn_width, name='BifpnInit')
    fpn_feature_maps = bifpn_layer(fpn_init_feature_maps, params.bifpn_width, name='Bifpn1')

    n_classes = 28
    n_channels_normals = 3
    n_channels_cmap = 1
    n_channels_contours = 1
    n_channels_out = n_classes + n_channels_normals + n_channels_cmap + n_channels_contours
    print(f'Ouput channels: {n_channels_out}')
    features_out = final_upscale(fpn_feature_maps, n_channels_out, name='FinalUpscale')
    
    model = tf.keras.models.Model(inputs=[image_input], outputs=features_out)
    
    print(model.summary())
    
    # for w in model.weights:
    #     print(w.name)

