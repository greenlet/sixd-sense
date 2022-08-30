import math
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, models
from typing import Tuple, Union, Callable, Optional

from sds.utils.common import IntOrTuple, int_to_tuple


class RotHeadType(Enum):
    Conv2d = 'conv2d'
    Conv3d = 'conv3d'


ROT_HEAD_TYPE_VALUES = [ht.value for ht in RotHeadType]
Activation = Callable[[tf.Tensor], tf.Tensor]


def block_down(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5,
               act: Optional[Activation] = tf.nn.relu) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in * strides[0] * strides[1]
    x = layers.Conv2D(ch_out, kernel_size, strides, 'same')(x)
    x = layers.BatchNormalization()(x)
    if act is not None:
        x = act(x)
    return x, ch_out


def block_up(x: tf.Tensor, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 4,
             act: Optional[Activation] = tf.nn.relu) -> Tuple[tf.Tensor, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in // (strides[0] * strides[1])
    x = layers.Conv2DTranspose(ch_out, kernel_size, strides, 'same')(x)
    if act is not None:
        x = act(x)
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
    x, ch = layers.Conv2D(1000, 4, activation='relu')(x), 1000
    x = layers.Dense(1000)(x)
    print('!!!', x)
    x_pos = tf.reshape(x, (-1, ch))

    x = activations.relu(x)
    x = layers.Dense(125000, activation='relu')(x)
    x = tf.reshape(x, (-1, 50, 50, 50, 1))
    x = layers.Conv3DTranspose(1, 8, 2, 'same', activation='sigmoid')(x)
    print('!!!', x)
    x = tf.reshape(x, (-1, 100, 100, 100))
    out_rot = x

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1006
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


def build_pose_layers_(input_size: int = 256, input_channels: int = 6, inp_pose_size: int = 6) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
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
    x, ch = layers.Conv2D(1000, 4, activation='relu')(x), 1000
    x = layers.Dense(1000)(x)
    print('!!!', x)
    x_pos = tf.reshape(x, (-1, ch))

    x = activations.relu(x)
    x = tf.reshape(x, (-1, 10, 10, 10, 1))
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='relu')(x), 1 # 20^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='relu')(x), 1 # 40^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='relu')(x), 1 # 80^3, 1
    x, ch = layers.Conv3DTranspose(1, 4, 2, padding='same', activation='sigmoid')(x), 1 # 160^3, 1
    print('!!!', x)
    out_rot = tf.reshape(x, (-1, 160, 160, 160))

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1006
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp, inp_pose), (out_rot, out_pos)


def is_pow_2(n: int) -> bool:
    while n > 1:
        if n % 2: return False
        n //= 2
    return True


def hbit(n: int) -> int:
    return int(np.log2(n))


def build_rot_head_2d(x: tf.Tensor, ch_in: int, out_size: int) -> tf.Tensor:
    nbits_in = hbit(ch_in)
    nbits_out = hbit(out_size)
    assert nbits_out <= nbits_in <= 2 * nbits_out

    ch = ch_in
    ker_sz = 2
    for nb in range(nbits_out):
        ch_out = ch if ch == out_size else ch // 2
        act = None if nb == nbits_out - 1 else tf.nn.relu
        x, ch = block_up(x, 2, ch, ch_out, ker_sz, act)
        ker_sz = 4

    return x


def build_rot_head_3d(x: tf.Tensor, ch_in: int, out_size: int) -> tf.Tensor:
    nbits_out = hbit(out_size)
    print('out_size:', out_size, 'nbits_out:', nbits_out)
    nb_in = hbit(ch_in)
    assert nb_in <= 2 * nbits_out
    ch = ch_in
    for nb_out in range(nbits_out):
        if nb_in:
            # bsub = 2 if nb_in + nb_out > nbits_out else 1
            bsub = min(nb_in, 2)
            ch //= 2**bsub
            nb_in -= bsub
        act = None if nb_out == nbits_out - 1 else 'relu'
        ker_sz, stride = 2, 2
        # if nb_out > 3:
        #     ker_sz, stride = 4, 4
        x = layers.Conv3DTranspose(ch, ker_sz, stride, activation=act)(x)
        print(x)

    x = tf.reshape(x, (-1, out_size, out_size, out_size))
    return x


def build_pose_layers_new(head_type: RotHeadType, inp_img_size: int = 256, out_cube_size: int = 128, inp_channels: int = 6, inp_pose_size: int = 6) -> Tuple[
        Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    assert is_pow_2(inp_img_size), f'input_size = {inp_img_size} is not power of 2'
    assert is_pow_2(out_cube_size), f'output_size = {out_cube_size} is not power of 2'
    assert 32 <= inp_img_size <= 512
    assert 32 <= out_cube_size <= 256
    nbits_in = hbit(inp_img_size)
    nbits_out = hbit(out_cube_size)
    input_shape = inp_img_size, inp_img_size, inp_channels
    inp_maps = tf.keras.Input(input_shape)
    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)

# Basic scenario:
#    size  channels
# 0   256         6
# 1   128         8
# 2    64        16
# 3    32        32
# 4    16        64
# 5     8       128
# 6     4       256
# 7     2       512
# 8     1      1024

    ch, ker_sz, stride = 4, 5, 2
    if nbits_in < 8:
        ch *= 2**(8 - nbits_in)
    elif nbits_in > 8:
        ker_sz, stride = 7, 4

    x = inp_maps
    while nbits_in > 1:
        act = None if nbits_in == 1 else tf.nn.relu
        x, ch = block_down(x, stride, ch, ch * 2, ker_sz, act)
        nbits_in -= hbit(stride)
        if nbits_in + hbit(ch) == 10:
            ker_sz, stride = 5, 2
        if nbits_in <= 4:
            ker_sz = 3
    ch *= 2
    x = layers.Conv2D(ch, 2, 1, 'valid')(x)

    print('Encoder:', x)
    x_pos = tf.reshape(x, (-1, ch))

    x = tf.nn.relu(x)
    if head_type == RotHeadType.Conv2d:
        x = build_rot_head_2d(x, ch, out_cube_size)
    elif head_type == RotHeadType.Conv3d:
        x = tf.expand_dims(x, -2)
        x = build_rot_head_3d(x, ch, out_cube_size)

    out_rot = tf.nn.sigmoid(x)
    print('Rot Head:', out_rot)

    x_pos = layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1024 + 6
    x_pos = layers.Dense(512, activation='relu')(x_pos)
    x_pos = layers.Dense(256, activation='relu')(x_pos)
    x_pos = layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp_maps, inp_pose), (out_rot, out_pos)


if __name__ == '__main__':
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 128, 128
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 256, 128
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 256, 256
    head_type, in_sz, out_sz = RotHeadType.Conv3d, 128, 128
    head_type, in_sz, out_sz = RotHeadType.Conv3d, 256, 128
    # head_type, in_sz, out_sz = RotHeadType.Conv3d, 256, 256
    inp, out = build_pose_layers_new(head_type, in_sz, out_cube_size=out_sz)
    print('out:', out)
    model = models.Model(inputs=inp, outputs=out)
    print(model.summary())

