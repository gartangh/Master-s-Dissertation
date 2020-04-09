import numpy as np
from numpy.random import randn
import nvtx.plugins.tf as nvtx_tf
import tensorflow as tf
from tensorflow.keras.losses import MAE
from tensorflow.keras.application import VGG19


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled


@nvtx_tf.ops.trace(message='Model', domain_name='Forward',
                   grad_domain_name='Gradient', enabled=True, trainable=True)
def profile(test):
    return model.predict(test)


def benchmark(batchsize=64):
    model = VGG19()
    model.compile(optimizer='adam', loss=MAE)
    model.summary()
    test = tf.convert_to_tensor(np.array(randn(*(batchsize, 224, 224, 3)), dtype=np.float32))

    # warmup
    profile(test)

    profile(test)
