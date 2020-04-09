import numpy as np
from numpy.random import randn
import tensorflow as tf


def channel_shuffle1(ngroups, input):
    n, h, w, c = input.shape.as_list()
    input_reshaped = tf.reshape(input, [-1, h, w, ngroups, c // ngroups])
    input_transposed = tf.transpose(input_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(input_transposed, [-1, h, w, c])
    return output


# def channel_shuffle2(ngroups, input):
#     input = tf.transpose(input, perm=[0, 3, 1, 2])
#     in_shape = input.get_shape().as_list()
#     in_channel = in_shape[1]
#     l = tf.reshape(input, [-1, ngroups, in_channel // ngroups] + in_shape[-2:])
#     l = tf.transpose(l, [0, 2, 1, 3, 4])
#     l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
#     l = tf.transpose(l, perm=[0, 2, 3, 1])
#     return l

print("1")
# test = np.array(randn(*(16, 7, 7, 256)), dtype=np.float32)
# test = np.arange(1, 16 * 7 * 7 * 256 + 1, 1).reshape((16, 7, 7, 256))
# test = np.arange(1, 1 * 1 * 1 * 4 + 1, 1).reshape((1, 1, 1, 4))
# test = np.arange(1, 1 * 1 * 1 * 9 + 1, 1).reshape((1, 1, 1, 9))
test = np.arange(1, 1 * 1 * 1 * 8 + 1, 1).reshape((1, 1, 1, 8))
output = channel_shuffle1(2, tf.convert_to_tensor(test))
print(output)

# print()
# print("2")
# test = np.arange(1, 1 * 1 * 1 * 8 + 1, 1).reshape((1, 1, 1, 8))
# output = channel_shuffle2(2, tf.convert_to_tensor(test))
# print(output)

