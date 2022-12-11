import numpy as np
import tensorflow as tf


def prefixer(prefix: str = ''):
    if not prefix:
        return lambda name: name
    return lambda name: f'{prefix}/{name}'


def is_pow_2(n: int) -> bool:
    while n > 1:
        if n % 2: return False
        n //= 2
    return True


def hbit(n: int) -> int:
    return int(np.log2(n))


def tf_img_to_float(img: tf.Tensor) -> tf.Tensor:
    return (tf.cast(img, tf.float32) - 127.5) / 127.5


def tf_float_to_img(t: tf.Tensor) -> tf.Tensor:
    t = tf.multiply(t, 127.5) + 127.5
    tf.clip_by_value(t, 0, 255)
    return tf.cast(t, tf.int8)


def np_img_to_float(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) - 127.5) / 127.5


def np_float_to_img(img: np.ndarray) -> np.ndarray:
    img = img * 127.5 + 127.5
    return np.clip(img, 0, 255).astype(np.uint8)

