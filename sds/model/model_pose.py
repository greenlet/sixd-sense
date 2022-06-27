import math

import tensorflow as tf
from typing import Tuple, Union

from sds.utils.common import IntOrTuple, int_to_tuple


def block_down(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in * strides[0] * strides[1]
    x = tf.keras.layers.Conv2D(ch_out, kernel_size, strides, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x, ch_out


def block_up(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in // (strides[0] * strides[1])
    x = tf.keras.layers.Conv2DTranspose(ch_out, kernel_size, strides, 'same')(x)
    return x, ch_out


def build_pose_layers(input_size: Tuple[int, int] = (128, 128), input_channels: int = 6) -> Tuple[tf.Tensor, tf.Tensor]:
    input_shape = input_size[1], input_size[0], input_channels
    inp = tf.keras.Input(input_shape)
    x = inp # 128, 128, 6
    x, ch = block_down(x, (2, 1), input_channels, 16) # 64, 128, 16
    x, ch = block_down(x, 2, ch) # 32, 64, 64
    x, ch = block_down(x, 2, ch) # 16, 32, 256
    x, ch = block_down(x, 2, ch) # 8, 16, 1024
    x, ch = block_down(x, 2, ch, ch * 2) # 4, 8, 2048
    x, ch = block_up(x, 2, ch, ch // 2) # 8, 16, 1024
    x, ch = block_up(x, 2, ch, ch // 2) # 16, 32, 512
    x, ch = block_up(x, 2, ch, ch // 2) # 32, 64, 256
    x, ch = block_up(x, 2, ch, ch // 2) # 64, 128, 128
    out = x
    return inp, out


def build_pose_layers_light(input_size: Tuple[int, int] = (128, 128), input_channels: int = 6) -> Tuple[tf.Tensor, tf.Tensor]:
    input_shape = input_size[1], input_size[0], input_channels
    inp = tf.keras.Input(input_shape)
    x, ch = inp, input_channels # 128, 128, 6
    x, ch = block_down(x, 2, ch, 16) # 64, 64, 16
    x, ch = block_down(x, 2, ch, ch * 2) # 32, 32, 32
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 16, 16, 64
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 8, 8, 128
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 4, 4, 256
    x, ch = block_down(x, (2, 1), ch, ch * 2, (3, 1)) # 2, 4, 512
    x, ch = block_up(x, 2, ch, ch // 2, 3) # 4, 8, 256
    x, ch = block_up(x, 2, ch, ch // 2, 3) # 8, 16, 128
    x, ch = block_up(x, 2, ch, ch, 3) # 16, 32, 128
    x, ch = block_up(x, 2, ch, ch, 3) # 32, 64, 128
    x, ch = block_up(x, 2, ch, ch, 3) # 64, 128, 128
    out = x
    return inp, out


if __name__ == '__main__':
    inp, out = build_pose_layers()
    inp, out = build_pose_layers_light()
    print('out:', out)
    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    print(model.summary())

