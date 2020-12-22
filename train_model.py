# Created By liguang
# Date: 2020/12/22
# ---------------------

import tensorflow as tf
import numpy as np


class Config(object):
    DEEP_HIDDEN_UNITS = [1024, 512, 256, 1]
    BATCH_SIZE = 32
    INPUT_TENSOR_DIM = 2048


C = Config()


def create_mlp():
    inputs_layer = tf.keras.layers.Input(shape=(C.INPUT_TENSOR_DIM,), name='inputs_layer', dtype="float32")

    deep = inputs_layer
    for unit in C.DEEP_HIDDEN_UNITS:
        deep = tf.keras.layers.BatchNormalization()(deep)
        deep = tf.keras.layers.Dense(unit, activation='relu')(deep)

    model = tf.keras.models.Model(inputs=[inputs_layer], outputs=deep, name='mlp_model')
    return model


def data_generator():
    dataset = tf.data.Dataset.from_tensor_slices((np.random.uniform(size=(C.BATCH_SIZE, C.INPUT_TENSOR_DIM)),
                                                  np.random.uniform(size=(C.BATCH_SIZE))))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(C.BATCH_SIZE)
    return dataset


def main():
    data = data_generator()
    model = create_mlp()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(data)

    infer_batch_size = 100
    for tensor in model.inputs:
        tensor.set_shape([infer_batch_size] + list(tensor.shape)[1:])

    model.save("./saved_model")



if __name__ == '__main__':
    main()
