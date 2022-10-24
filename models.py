import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


def get_model_generator(args):
    name = args.model
    if name == 'baseline':
        model = ModelGenerator(args)
    else:
        raise ValueError(
            '{0} is not a valid model_type.'.format(args.model_type))
    return model


class ModelGenerator(object):
    def __init__(self, args):
        self.args = args

    def get_output_layer(self, x, name):
        return Dense(2, activation='softmax', name=name)(x)

    def get_main_model(self, x):
        x = tf.keras.layers.Embedding(2, 8)(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(4):
            x = Dense(16, activation='relu')(x)
        return x, x

    def get_structure(self):
        inputs = Input(shape=(3,), dtype=tf.int32)
        x1, x2 = self.get_main_model(inputs)

        outputs = [
            self.get_output_layer(x1, 'y1'),
            self.get_output_layer(x2, 'y2')
        ]
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        model = self.get_structure()
        adam = tf.keras.optimizers.Adam(lr=self.args.lr)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
