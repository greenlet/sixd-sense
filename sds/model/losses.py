from typing import Tuple

import numpy as np
import tensorflow as tf

from sds.model.processing import tf_img_to_float


class MseNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = tf_img_to_float(y_true)
        diff = (y_true - y_pred) ** 2
        res = tf.reduce_mean(diff)
        return res


class CosNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = tf_img_to_float(y_true)
        y_true, y_pred = tf.reshape(y_true, (-1, 3)), tf.reshape(y_pred, (-1, 3))
        loss_cos = tf.reduce_sum(y_true * y_pred, axis=-1)
        y_pred_norm = tf.norm(y_pred, axis=-1)
        loss_cos = loss_cos / y_pred_norm + 1
        loss_norm = 0.05 * (y_pred_norm - 1) ** 2
        res = tf.reduce_mean(loss_cos + loss_norm)
        return res


class SparseCategoricalCrossEntropyNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true, tf.bool)[..., 0]
        y_true, y_pred = y_true[mask], y_pred[mask]
        return self.loss(y_true, y_pred)


def get_rot_prob(y_pred: tf.Tensor, inds: tf.Tensor) -> tf.Tensor:
    return tf.map_fn(lambda pi: tf.gather_nd(pi[0], tf.reshape(pi[1], (1, 3))), (y_pred, inds), fn_output_signature=tf.TensorSpec((1,), dtype=tf.float32))


class RotVecLoss(tf.keras.losses.Loss):
    def __init__(self, N: int):
        super().__init__(name=self.__class__.__name__)
        self.N = N

    """Calculates loss between GT Rodriguez vector and predicted probabilities of all possible vectors

    Finds y_true indices in N x N x N cube and penalizes corresponding y_pred value

    :param y_true: batch_sz x 3 tensor containing Rodriguez rotation vectors
    :param y_pred: batch_sz x N x N x N tensor containing predicted Rodriguez vectors probabilities
    """
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ys = y_true / (2 * np.pi)
        y1 = ys + 1 / 2
        y2 = -ys + 1 / 2
        y1 = tf.cast(tf.math.floor(y1 * self.N), tf.int64)
        y2 = tf.cast(tf.math.floor(y2 * self.N), tf.int64)

        # p1 = tf.gather_nd(y_pred, y1, batch_dims=1)
        # p2 = tf.gather_nd(y_pred, y2, batch_dims=1)

        p1 = get_rot_prob(y_pred, y1)
        p2 = get_rot_prob(y_pred, y2)
        res = -tf.reduce_mean(tf.math.log(p1) + tf.math.log(p2))
        return res


