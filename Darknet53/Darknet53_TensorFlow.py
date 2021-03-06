import time
from timeit import timeit

import numpy as np
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, LeakyReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.losses import MAE
from tensorflow.keras.models import Model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.optimizer.set_jit(True)  # XLA enabled


def Darknet53():
    i = Input(shape=(256, 256, 3))

    # 1-2
    x = Conv2D(32, (3, 3), padding='same', strides=(1, 1))(i)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))
    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))

    # 3-4
    y = Conv2D(32, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)

    # 5
    x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))

    # 6-9
    y = Conv2D(64, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(64, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(128, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)

    # 10
    x = Conv2D(256, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))

    # 11-26
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(128, (1, 1), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(256, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)

    # 27
    x = Conv2D(512, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))

    # 28-43
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(256, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(512, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)

    # 44
    x = Conv2D(1024, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(x))

    # 45-52
    y = Conv2D(512, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(1024, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(512, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(1024, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(512, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(1024, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)
    y = Conv2D(512, (1, 1), padding='same', strides=(1, 1))(x)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    y = Conv2D(1024, (3, 3), padding='same', strides=(1, 1))(y)
    y = LeakyReLU(alpha=0.01)(BatchNormalization(axis=3)(y))
    x = tf.math.add(x, y)

    # 53
    # Global Mean Pooling layer
    x = GlobalAveragePooling2D()(x)
    # Fully connected layer
    x = Dense(1000, activation='softmax')(x)

    model = Model(i, x)

    return model


m = Darknet53()
m.compile(optimizer='adam', loss=MAE)


def benchmark_tensorflow(batchsize):
    ip = tf.convert_to_tensor(np.array(randn(*(batchsize, 256, 256, 3)), dtype=np.float32))

    # warm-up
    m.predict(ip)

    time.sleep(10)

    # benchmark
    print(timeit(lambda: m.predict(ip), number=10))
