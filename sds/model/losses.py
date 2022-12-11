from typing import Tuple

import numpy as np
import tensorflow as tf

from sds.model.utils import tf_img_to_float

logger = tf.get_logger()


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
    def __init__(self, N: int, diff_threshold: float = 0.2):
        super().__init__(name=self.__class__.__name__)
        self.N = tf.constant(N, tf.float32)
        self.diff_thres = tf.constant(diff_threshold, tf.float32)

    def _rot_vec_to_inds(self, rv: tf.Tensor) -> tf.Tensor:
        ys = (rv + np.pi) / (2 * np.pi)
        inds = tf.clip_by_value(tf.math.floor(ys * self.N), 0, self.N - 1)
        return tf.cast(inds, tf.int64)

    """Calculates loss between GT Rodriguez vector and predicted probabilities of all possible vectors

    Finds y_true indices in N x N x N cube and penalizes corresponding y_pred value

    :param y_true: batch_sz x 3 tensor containing Rodriguez rotation vectors
    :param y_pred: batch_sz x N x N x N tensor containing predicted Rodriguez vectors probabilities
    """
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        i, j = tf.cast(y_true[:, 0], tf.int64), tf.cast(y_true[:, 1], tf.int64)
        rvec, diff = y_true[:, 2:5], y_true[:, 5]
        m_pos = i == j
        i_pos, rv_pos = i[m_pos][..., None], rvec[m_pos]
        inds = self._rot_vec_to_inds(rv_pos)
        inds = tf.concat([i_pos, inds], axis=1)
        p = tf.gather_nd(y_pred, inds)
        res = -tf.reduce_mean(tf.math.log(p))

        m_neg = (diff > self.diff_thres) & ~m_pos

        def calc_neg():
            i_neg, rv_neg = i[m_neg][..., None], rvec[m_neg]
            inds = self._rot_vec_to_inds(rv_neg)
            inds = tf.concat([i_neg, inds], axis=1)
            p = tf.gather_nd(y_pred, inds)
            return -tf.reduce_mean(tf.math.log(1 - p))

        res += tf.cond(tf.reduce_sum(tf.cast(m_neg, tf.int32)) > 0, calc_neg, lambda: tf.constant(0, tf.float32))
        return res


class TransLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = y_true - y_pred
        diff = tf.reduce_sum(diff * diff, axis=-1)
        diff = tf.reduce_mean(diff)
        return diff


class RotSphereLoss(tf.keras.losses.Loss):
    def __init__(self, graph_pts: tf.Tensor, n_levels: int, diff_threshold: float = 0.2):
        super().__init__(name=self.__class__.__name__)
        self.graph_pts = graph_pts
        self.n_levels = n_levels
        self.diff_thres = tf.constant(diff_threshold, tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        i, j = tf.cast(y_true[:, 0], tf.int64), tf.cast(y_true[:, 1], tf.int64)
        rvec, diff = y_true[:, 2:5], y_true[:, 5]
        ang = tf.norm(rvec, axis=-1, keepdims=True)
        rv = rvec / ang
        m_pos = i == j
        i_pos, rv_pos, ang_pos = i[m_pos][..., None], rv[m_pos], ang[m_pos]

        inds = self._rot_vec_to_inds(rv_pos)
        inds = tf.concat([i_pos, inds], axis=1)
        p = tf.gather_nd(y_pred, inds)
        res = -tf.reduce_mean(tf.math.log(p))

        m_neg = (diff > self.diff_thres) & ~m_pos

        def calc_neg():
            i_neg, rv_neg = i[m_neg][..., None], rvec[m_neg]
            inds = self._rot_vec_to_inds(rv_neg)
            inds = tf.concat([i_neg, inds], axis=1)
            p = tf.gather_nd(y_pred, inds)
            return -tf.reduce_mean(tf.math.log(1 - p))

        res += tf.cond(tf.reduce_sum(tf.cast(m_neg, tf.int32)) > 0, calc_neg, lambda: tf.constant(0, tf.float32))
        return res

