import tensorflow as tf


def img_to_float(img: tf.Tensor) -> tf.Tensor:
    return (tf.cast(img, tf.float32) - 127.5) / 127.5


def float_to_img(t: tf.Tensor) -> tf.Tensor:
    t = tf.multiply(t, 127.5) + 127.5
    tf.clip_by_value(t, 0, 255)
    return tf.cast(t, tf.int8)


