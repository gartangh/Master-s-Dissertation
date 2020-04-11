import os

import numpy as np
import nvtx.plugins.tf as nvtx_tf
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras.application import VGG19
from tensorflow.keras.losses import MAE

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled


@nvtx_tf.ops.trace(message='Model', domain_name='Forward',
                   grad_domain_name='Gradient', enabled=True, trainable=True)
def profile(m, ip):
    return m.predict(ip)


def benchmark(batchsize=256):
    m = VGG19()
    m.compile(optimizer='adam', loss=MAE)
    m.summary()
    ip = tf.convert_to_tensor(np.array(randn(*(batchsize, 224, 224, 3)), dtype=np.float32))

    # warmup
    profile(m, ip)

    profile(m, ip)


if __name__ == '__main__':
    benchmark(4)
