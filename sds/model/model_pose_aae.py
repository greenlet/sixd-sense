from typing import Tuple, List, Dict, Any, Optional, Union

import tensorflow as tf

from sds.model.utils import hbit


def block_down(x: Any, ch_out: int, ker_sz: int = 5, strides: int = 2) -> Any:
    x = tf.keras.layers.Conv2D(ch_out, ker_sz, strides, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x


def block_up(x: Any, ch_out: int, ker_sz: int = 4, strides: int = 2, act: Optional[Any] = tf.nn.relu) -> Any:
    x = tf.keras.layers.Conv2DTranspose(ch_out, ker_sz, strides, 'same', activation=act)(x)
    return x


def build_aae_layers(img_size: int = 256, inout_channels: int = 6, latent_space_size: int = 128, batch_size: Optional[int] = None) -> Tuple[tf.keras.Input, Any]:
    input_shape = img_size, img_size, inout_channels
    inp = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)

    x = inp
    x = block_down(x, 64) # 256 / 2 = 128
    x = block_down(x, 128) # 128 / 2 = 64
    x = block_down(x, 128) # 64 / 2 = 32
    x = block_down(x, 256) # 32 / 2 = 16
    x = block_down(x, 512) # 16 / 2 = 8

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_space_size)(x)
    z = x

    hb = hbit(img_size)
    sz = 2 ** (hb - 5)
    x = tf.keras.layers.Dense(sz * sz * 512, tf.nn.relu)(x)
    x = tf.reshape(x, (-1, sz, sz, 512))
    x = block_up(x, 256)
    x = block_up(x, 128)
    x = block_up(x, 128)
    x = block_up(x, 64)
    x = block_up(x, inout_channels, act=tf.nn.sigmoid)

    return inp, x


class AaeLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int, img_size: int, channels: int, bootstrap_ratio: int = 1):
        super().__init__(name=self.__class__.__name__)
        self.bootstrap_ratio = bootstrap_ratio
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.ch_half = self.channels // 2
        self.k = self.img_size * self.img_size * self.ch_half // self.bootstrap_ratio

    def _top_k_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        diff = tf.math.squared_difference(y_true, y_pred)
        diff = tf.reduce_mean(diff, axis=-1)
        diff = tf.reshape(diff, (self.batch_size, self.img_size * self.img_size))
        diff, _ = tf.nn.top_k(diff, self.k, sorted=False)
        return tf.reduce_mean(diff)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if self.bootstrap_ratio > 1:
            y_true_1, y_true_2 = y_true[..., :self.ch_half], y_true[..., self.ch_half:]
            y_pred_1, y_pred_2 = y_pred[..., :self.ch_half], y_pred[..., self.ch_half:]
            loss_1 = self._top_k_loss(y_true_1, y_pred_1)
            loss_2 = self._top_k_loss(y_true_2, y_pred_2)
            loss = loss_1 + loss_2
        else:
            diff = tf.math.squared_difference(y_true, y_pred)
            loss = tf.reduce_mean(diff)

        return loss


def test_aae():
    inp, out = build_aae_layers(img_size=128, batch_size=10)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    print(model.summary())


if __name__ == '__main__':
    test_aae()

