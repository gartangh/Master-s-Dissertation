import time
from timeit import timeit

import numpy as np
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras.layers import BatchNormalization, Conv2D, LeakyReLU, GlobalAveragePooling2D, MaxPool2D, Softmax, Flatten
from tensorflow.keras.losses import MAE
from tensorflow.keras.models import Sequential

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled


Darknet19 = Sequential([
    # 1
    Conv2D(32, (3, 3), padding='same', strides=(1, 1), input_shape=(224, 224, 3)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    MaxPool2D(pool_size=(2, 2), padding='same', strides=(2, 2)),
    # 2
    Conv2D(64, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    MaxPool2D(pool_size=(2, 2), padding='same', strides=(2, 2)),
    # 3-5
    Conv2D(128, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(64, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(128, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    MaxPool2D(pool_size=(2, 2), padding='same', strides=(2, 2)),
    # 6-8
    Conv2D(256, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(128, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(256, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    MaxPool2D(pool_size=(2, 2), padding='same', strides=(2, 2)),
    # 9-13
    Conv2D(512, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(256, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(512, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(256, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(512, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    MaxPool2D(pool_size=(2, 2), padding='same', strides=(2, 2)),
    # 14-18
    Conv2D(1024, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(512, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(1024, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(512, (3, 3), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    Conv2D(1024, (1, 1), padding='same', strides=(1, 1)),
    BatchNormalization(), LeakyReLU(alpha=0.01),
    # 19
    Conv2D(1000, (1, 1), padding='same', strides=(1, 1)),

    GlobalAveragePooling2D(), # Global Mean Pooling layer
    Flatten(), # Flattening operation
    Softmax(), # Softmax activation
])


def benchmark_tensorflow(batchsize):
    gm = Darknet19
    gm.compile(optimizer='adam', loss=MAE)

    gip = tf.convert_to_tensor(np.array(randn(*(batchsize, 224, 224, 3)), dtype=np.float32))

    # warm-up
    gm.predict(gip)

    for _ in range(10):
        time.sleep(1)
        gm.predict(gip)

    time.sleep(10)

    # benchmark
    print(timeit(lambda: gm.predict(gip), number=10))
