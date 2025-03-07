import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


def get_model_generator(args):
    name = args.model
    if name == 'baseline':
        model = BaselineModelGenerator(args)
    elif name == 'proposed':
        model = ProposedModelGenerator(args)
    elif name == 'no_regularization':
        model = NoRegularizationModelGenerator(args)
    elif name == 'no_decoder':
        model = NoDecoderModelGenerator(args)
    else:
        raise ValueError(
            '{0} is not a valid model.'.format(args.model))
    return model


class AbstractModelGenerator(object):
    def __init__(self, args):
        self.args = args

    def get_output_layer(self, x, name):
        return Dense(2, activation='softmax', name=name)(x)

    def get_structure(self):
        inputs = Input(shape=(3,), dtype=tf.int64)
        outputs, hidden = self.get_main_model(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        encoder = Model(inputs=inputs, outputs=hidden)
        return model, encoder

    def get_model(self):
        model, encoder = self.get_structure()
        adam = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, encoder

    def ff(self, out_size, x, activation, depth=3):
        for _ in range(depth - 1):
            x = Dense(4 * self.args.embedding_size, activation='relu')(x)
        return Dense(out_size, activation=activation)(x)

    def encode_factor(self, x):
        x = tf.keras.layers.Embedding(2, self.args.embedding_size)(x)
        return x

    def get_main_model(self, x):
        y1, y2 = self.decoder(x)
        return [y1, y2], x

    # baseline ---------------------------------------
    def baseline_regularization(self, x):
        return x

    def baseline_decoder(self, x):
        x = self.encode_factor(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.ff(4 * self.args.embedding_size, x, 'relu', depth=2)
        y1 = self.get_output_layer(x, 'y1')
        y2 = self.get_output_layer(x, 'y2')
        return y1, y2

    # proposed ---------------------------------------
    def proposed_regularization(self, x):
        x = tf.keras.layers.ActivityRegularization(l2=self.args.beta)(x)
        x = tf.keras.layers.GaussianNoise(self.args.alpha)(x)
        return x

    def attention_decoder(self, x):
        attention = self.encode_factor(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = self.ff(3, attention, 'softmax', depth=2)
        attention = tf.expand_dims(attention, 1)

        values = self.encode_factor(x)
        attended = tf.matmul(attention, values)
        attended = tf.keras.layers.Flatten()(attended)
        y = self.ff(2, attended, 'softmax', depth=2)
        return y, y

    def xor_decoder(self, x):
        x = self.encode_factor(x)
        h, x3 = tf.split(x, [2, 1], 1)
        h = tf.keras.layers.Flatten()(h)
        h = self.ff(2 * self.args.embedding_size, h, 'linear', depth=2)
        h = self.regularization(h)
        x3 = tf.keras.layers.Flatten()(x3)
        y = tf.concat([h, x3], -1)
        y = self.ff(2, y, 'softmax', depth=2)
        return y, y

    def proposed_decoder(self, x):
        if self.args.task == "attention":
            return self.attention_decoder(x)
        elif self.args.task == "xor":
            return self.xor_decoder(x)
        assert False


class BaselineModelGenerator(AbstractModelGenerator):
    def regularization(self, x):
        return self.baseline_regularization(x)

    def decoder(self, x):
        return self.baseline_decoder(x)


class ProposedModelGenerator(AbstractModelGenerator):
    def regularization(self, x):
        return self.proposed_regularization(x)

    def decoder(self, x):
        return self.proposed_decoder(x)


class NoRegularizationModelGenerator(ProposedModelGenerator):
    def regularization(self, x):
        return self.baseline_regularization(x)


class NoDecoderModelGenerator(ProposedModelGenerator):
    def decoder(self, x):
        return self.baseline_decoder(x)
