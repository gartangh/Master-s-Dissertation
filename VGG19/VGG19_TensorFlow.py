import numpy as np
import nvtx.plugins.tf as nvtx_tf
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MAE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled

m = VGG19()
m.compile(optimizer='adam', loss=MAE)
m.summary()


@nvtx_tf.ops.trace(message='VGG19 TensorFlow', domain_name='Forward',
                   grad_domain_name='Gradient', enabled=True, trainable=True)
def profile(input):
    return m.predict(input)


def benchmark(batchsize):
    ip = np.array(randn(*(batchsize, 224, 224, 3)), dtype=np.float32)

    # warmup
    profile(tf.convert_to_tensor(ip))

    profile(tf.convert_to_tensor(ip))


if __name__ == '__main__':
    benchmark(1)
