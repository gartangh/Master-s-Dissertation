import time
from timeit import timeit

import numpy as np
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MAE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled

m = VGG19()
m.compile(optimizer='adam', loss=MAE)


def benchmark(batchsize):
    ip = tf.convert_to_tensor(np.array(randn(*(batchsize, 224, 224, 3)), dtype=np.float32))

    # warmup
    m.predict(ip)

    time.sleep(10)

    # benchmark
    print(timeit(lambda: m.predict(ip), number=10))
