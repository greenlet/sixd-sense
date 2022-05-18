import tensorflow as tf

from sds.model.processing import img_to_float


class MseNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = img_to_float(y_true)
        diff = (y_true - y_pred) ** 2
        res = tf.reduce_mean(diff)
        return res


class CosNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = img_to_float(y_true)
        y_true, y_pred = tf.reshape(y_true, (-1, 3)), tf.reshape(y_pred, (-1, 3))
        loss_cos = tf.reduce_sum(y_true * y_pred, axis=-1) + 1
        y_pred_norm = tf.norm(y_pred, axis=-1)
        loss_cos /= y_pred_norm
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


