import os

import tensorflow as tf


CUDA_ENV_NAME = 'CUDA_VISIBLE_DEVICES'


def tf_set_gpu_incremental_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def tf_set_use_device(cpu: bool = True):
    if cpu:
        os.environ[CUDA_ENV_NAME] = '-1'
        # physical_devices = tf.config.list_physical_devices('CPU')
        # tf.config.set_visible_devices(physical_devices, 'CPU')
    else:
        os.environ[CUDA_ENV_NAME] = '0'
        tf_set_gpu_incremental_memory_growth()


