from enum import Enum

import tensorflow as tf
from typing import Tuple, Callable, Optional, Any

from sds.model.utils import is_pow_2, hbit
from sds.utils.utils import IntOrTuple, int_to_tuple


class RotHeadType(Enum):
    Conv2d = 'conv2d'
    Conv3d = 'conv3d'


ROT_HEAD_TYPE_VALUES = [ht.value for ht in RotHeadType]
Activation = Callable[[tf.Tensor], tf.Tensor]


def block_down(x: Any, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 5,
               act: Any = tf.nn.relu) -> Tuple[Any, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in * strides[0] * strides[1]
    x = tf.keras.layers.Conv2D(ch_out, kernel_size, strides, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if act is not None:
        x = act(x)
    return x, ch_out


def block_up(x: Any, strides: IntOrTuple, ch_in: int, ch_out: int = 0, kernel_size: IntOrTuple = 4,
             act: Any = tf.nn.relu) -> Tuple[Any, int]:
    strides = int_to_tuple(strides)
    if not ch_out:
        ch_out = ch_in // (strides[0] * strides[1])
    x = tf.keras.layers.Conv2DTranspose(ch_out, kernel_size, strides, 'same')(x)
    if act is not None:
        x = act(x)
    return x, ch_out


def build_layers_down(inp_img_size: int = 256, inp_channels: int = 6, batch_size: Optional[int] = None) -> Tuple[Any, Any, int]:
    assert is_pow_2(inp_img_size), f'input_size = {inp_img_size} is not power of 2'
    assert 32 <= inp_img_size <= 512
    nbits_in = hbit(inp_img_size)
    input_shape = inp_img_size, inp_img_size, inp_channels
    inp = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)

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

    x = inp
    while nbits_in > 1:
        act = None if nbits_in == 1 else tf.nn.relu
        x, ch = block_down(x, stride, ch, ch * 2, ker_sz, act)
        nbits_in -= hbit(stride)
        if nbits_in + hbit(ch) == 10:
            ker_sz, stride = 5, 2
        if nbits_in <= 4:
            ker_sz = 3
    ch *= 2
    x = tf.keras.layers.Conv2D(ch, 2, 1, 'valid')(x)

    out = tf.reshape(x, (-1, ch))

    return inp, out, ch


def build_rot_head_2d(x: Any, ch_in: int, out_size: int) -> Any:
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


def build_rot_head_3d(x: Any, ch_in: int, out_size: int) -> Any:
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
        if nb_out >= 4:
            ker_sz = 7
        elif nb_out >= 2:
            ker_sz = 5
        x = tf.keras.layers.Conv3DTranspose(ch, ker_sz, stride, activation=act, padding='same')(x)
        print(x)

    x = tf.reshape(x, (-1, out_size, out_size, out_size))
    return x


def build_pose_layers(head_type: RotHeadType, inp_img_size: int = 256, out_cube_size: int = 128, inp_channels: int = 6, inp_pose_size: int = 6) -> Tuple[
        Tuple[Any, Any], Tuple[Any, Any]]:
    assert is_pow_2(out_cube_size), f'output_size = {out_cube_size} is not power of 2'
    assert 32 <= out_cube_size <= 256

    inp_pose = tf.keras.Input((inp_pose_size,), dtype=tf.float32)
    inp_maps, x, ch = build_layers_down(inp_img_size, inp_channels)
    x_pos = x

    x = tf.nn.relu(x)
    if head_type == RotHeadType.Conv2d:
        x = build_rot_head_2d(x, ch, out_cube_size)
    elif head_type == RotHeadType.Conv3d:
        x = tf.expand_dims(x, -2)
        x = build_rot_head_3d(x, ch, out_cube_size)

    out_rot = tf.nn.sigmoid(x)
    print('Rot Head:', out_rot)

    x_pos = tf.keras.layers.Concatenate(axis=-1)([x_pos, inp_pose]) # 1024 + 6
    x_pos = tf.keras.layers.Dense(512, activation='relu')(x_pos)
    x_pos = tf.keras.layers.Dense(256, activation='relu')(x_pos)
    x_pos = tf.keras.layers.Dense(3)(x_pos)
    out_pos = x_pos

    return (inp_maps, inp_pose), (out_rot, out_pos)


def test_cnn():
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 128, 128
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 256, 128
    head_type, in_sz, out_sz = RotHeadType.Conv2d, 256, 256
    head_type, in_sz, out_sz = RotHeadType.Conv3d, 128, 128
    head_type, in_sz, out_sz = RotHeadType.Conv3d, 256, 128
    # head_type, in_sz, out_sz = RotHeadType.Conv3d, 256, 256
    inp, out = build_pose_layers(head_type, in_sz, out_cube_size=out_sz)
    print('out:', out)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    print(model.summary())


if __name__ == '__main__':
    test_cnn()

