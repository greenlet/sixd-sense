import math

import tensorflow as tf
from tensorflow.keras import layers, activations, models
from typing import Tuple, Union

from sds.utils.common import IntOrTuple, int_to_tuple


def block_down(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in * strides[0] * strides[1]
    x = layers.Conv2D(ch_out, kernel_size, strides, 'same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x, ch_out


def block_up(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in // (strides[0] * strides[1])
    x = layers.Conv2DTranspose(ch_out, kernel_size, strides, 'same')(x)
    return x, ch_out


# TODO: Generalize to various input sizes
def build_pose_layers_1(input_size: int = 128, input_channels: int = 6, inp_pose_size: int = 6) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    assert input_size == 128
    input_shape = input_size, input_size, input_channels
    inp = tf.keras.Input(input_shape)
    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)
    x, ch = inp, input_channels # 128, 128, 6
    x, ch = block_down(x, 2, ch, 16) # 64, 64, 16
    x, ch = block_down(x, 2, ch, ch * 2) # 32, 32, 32
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 16, 16, 64
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 8, 8, 128
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 4, 4, 256
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 2, 2, 512
    x, ch = layers.Conv2D(ch * 2, 2, activation='relu')(x), ch * 2 # 1, 1, 1024

    x_pos = tf.reshape(x, (-1, ch))

    x = tf.expand_dims(x, -2)
    x, ch = layers.Conv3DTranspose(256, 2, 2, activation='relu')(x), 256 # 2, 2, 256
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 4 # 4^3, 64
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 8^3, 32
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 16^3, 16
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 32^3, 8
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 64^3, 4
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 128^3, 2
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2)(x), ch // 2 # 256^3, 1
    x = tf.reshape(x, (-1, 256, 256, 256))
    x = activations.sigmoid(x)
    out_rot = x

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1027
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3, activation='relu')(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


def build_pose_layers_2(input_size: int = 256, input_channels: int = 6, inp_pose_size: int = 6) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    assert input_size == 256
    input_shape = input_size, input_size, input_channels
    inp = tf.keras.Input(input_shape)
    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)
    x, ch = inp, input_channels # 256, 256, 6
    x, ch = block_down(x, 2, ch, 8) # 128, 128, 8
    x, ch = block_down(x, 2, ch, ch * 2) # 64, 64, 16
    x, ch = block_down(x, 2, ch, ch * 2) # 32, 32, 32
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 16, 16, 64
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 8, 8, 128
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 4, 4, 256
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 2, 2, 512
    x, ch = layers.Conv2D(ch * 2, 2, activation='relu')(x), ch * 2 # 1, 1, 1024

    x_pos = tf.reshape(x, (-1, ch))

    x = tf.expand_dims(x, -2)
    x, ch = layers.Conv3DTranspose(256, 2, 2, activation='relu')(x), 256 # 2^3, 256
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 4 # 4^3, 64
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 2 # 8^3, 16
    x, ch = layers.Conv3DTranspose(ch // 4, 4, 4, activation='relu')(x), ch // 2 # 32^3, 4
    x, ch = layers.Conv3DTranspose(ch // 4, 4, 4, activation='sigmoid')(x), ch // 2 # 128^3, 1
    out_rot = tf.reshape(x, (-1, 128, 128, 128))

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1027
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


def build_pose_layers_3(input_size: int = 256, input_channels: int = 6, inp_pose_size: int = 6) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    assert input_size == 256
    input_shape = input_size, input_size, input_channels
    inp = tf.keras.Input(input_shape)
    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)
    x, ch = inp, input_channels # 256, 256, 6
    x, ch = block_down(x, 2, ch, 8) # 128, 128, 8
    x, ch = block_down(x, 2, ch, ch * 2) # 64, 64, 16
    x, ch = block_down(x, 2, ch, ch * 2) # 32, 32, 32
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 16, 16, 64
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 8, 8, 128
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 4, 4, 256
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 2, 2, 512
    x, ch = layers.Conv2D(ch * 4, 2, activation='relu')(x), ch * 4 # 1, 1, 2048

    x_pos = tf.reshape(x, (-1, ch))

    x = tf.expand_dims(x, -2)
    x, ch = layers.Conv3DTranspose(512, 2, 2, activation='relu')(x), 512 # 2, 2, 256
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 4 # 4^3, 64
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 2 # 8^3, 16
    x, ch = layers.Conv3DTranspose(ch // 4, 2, 2, activation='relu')(x), ch // 2 # 16^3, 8
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 32^3, 4
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2, activation='relu')(x), ch // 2 # 64^3, 2
    x, ch = layers.Conv3DTranspose(ch // 2, 2, 2)(x), ch // 2 # 128^3, 1
    x = tf.reshape(x, (-1, 128, 128, 128))
    x = activations.sigmoid(x)
    out_rot = x

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1027
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


def build_pose_layers(input_size: int = 256, input_channels: int = 6, inp_pose_size: int = 6) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    assert input_size == 256
    input_shape = input_size, input_size, input_channels
    inp = tf.keras.Input(input_shape)
    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)
    x, ch = inp, input_channels # 256, 256, 6
    x, ch = block_down(x, 2, ch, 8) # 128, 128, 8
    x, ch = block_down(x, 2, ch, ch * 2) # 64, 64, 16
    x, ch = block_down(x, 2, ch, ch * 2) # 32, 32, 32
    x, ch = block_down(x, 2, ch, ch * 2) # 16, 16, 64
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 8, 8, 128
    x, ch = block_down(x, 2, ch, ch * 2, 3) # 4, 4, 256
    x, ch = layers.Conv2D(1000, 4)(x), 1000
    print('!!!', x)
    x_pos = tf.reshape(x, (-1, ch))

    x = tf.reshape(x, (-1, 10, 10, 10, 1))
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='sigmoid')(x), 1 # 20^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='sigmoid')(x), 1 # 40^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='sigmoid')(x), 1 # 80^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='sigmoid')(x), 1 # 160^3, 1
    print('!!!', x)
    out_rot = tf.reshape(x, (-1, 160, 160, 160))

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1006
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


if __name__ == '__main__':
    inp, out = build_pose_layers()
    # inp, out = build_pose_layers_2()
    print('out:', out)
    model = models.Model(inputs=inp, outputs=out)
    print(model.summary())

