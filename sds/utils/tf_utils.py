import os
from typing import Optional

import tensorflow as tf


CUDA_ENV_NAME = 'CUDA_VISIBLE_DEVICES'


def tf_set_gpu_incremental_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                val = True
                print('GPU:', gpu, 'set_memory_growth', val)
                tf.config.experimental.set_memory_growth(gpu, val)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def tf_set_use_device_(device_id: str):
    os.environ[CUDA_ENV_NAME] = device_id
    if device_id != '-1':
        tf_set_gpu_incremental_memory_growth()


def tf_set_use_device(device_id: str):
    os.environ[CUDA_ENV_NAME] = device_id
    if device_id != '-1':
        devices = []
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                if gpu.name.endswith(device_id):
                    devices.append(gpu)
                    break
        devices += tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(devices)
        tf.config.experimental.set_memory_growth(devices[0], True)



