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
    elif name == 'no_encoder':
        model = NoEncoderModelGenerator(args)
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
        adam = tf.keras.optimizers.Adam(lr=self.args.lr)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, encoder

    def ff(self, out_size, x, activation, depth=3):
        for _ in range(depth - 1):
            x = Dense(128, activation='relu')(x)
        return Dense(out_size, activation=activation)(x)

    def get_main_model(self, x):
        # encoder
        h = self.encoder(x)

        # regularization
        r = self.regularization(h)

        # decoder
        y1, y2 = self.decoder(r)
        return [y1, y2], h

    # baseline ---------------------------------------
    def baseline_encoder(self, x):
        x = tf.keras.layers.Embedding(2, 64)(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.ff(2, x, 'linear', depth=8)
        return x

    def baseline_regularization(self, x):
        return x

    def baseline_decoder(self, x):
        x = self.ff(128, x, 'relu', depth=2)
        y1 = self.get_output_layer(x, 'y1')
        y2 = self.get_output_layer(x, 'y2')
        return y1, y2

    # proposed ---------------------------------------
    def get_a_decoder(self, x):
        x = tf.keras.layers.Embedding(2, 32)(x)
        x = self.regularization(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.ff(1, x, 'linear', depth=8)
        return x

    def proposed_encoder(self, x):
        h1 = self.get_a_decoder(x)
        h2 = self.get_a_decoder(x)
        h = tf.concat([h1, h2], -1)
        return h

    def proposed_regularization(self, x):
        x = tf.keras.layers.ActivityRegularization(l2=self.args.beta)(x)
        x = tf.keras.layers.GaussianNoise(self.args.alpha)(x)
        return x

    def proposed_decoder(self, x):
        h1, h2 = tf.split(x, 2, -1)
        h1 = self.ff(1, h1, 'sigmoid', depth=2)
        h2 = self.ff(1, h2, 'sigmoid', depth=2)
        y1 = tf.concat([h2, 1 - h2], -1)
        xor = 2 * (h1 - 0.5) * (h2 - 0.5) + 0.5
        y2 = tf.concat([xor, 1 - xor], -1)
        return y1, y2


class BaselineModelGenerator(AbstractModelGenerator):
    def encoder(self, x):
        return self.baseline_encoder(x)

    def regularization(self, x):
        return self.baseline_regularization(x)

    def decoder(self, x):
        return self.baseline_decoder(x)


class ProposedModelGenerator(AbstractModelGenerator):
    def encoder(self, x):
        return self.proposed_encoder(x)

    def regularization(self, x):
        return self.proposed_regularization(x)

    def decoder(self, x):
        return self.proposed_decoder(x)


class NoRegularizationModelGenerator(ProposedModelGenerator):
    def regularization(self, x):
        return self.baseline_regularization(x)


class NoEncoderModelGenerator(ProposedModelGenerator):
    def encoder(self, x):
        return self.baseline_encoder(x)


class NoDecoderModelGenerator(ProposedModelGenerator):
    def decoder(self, x):
        return self.baseline_decoder(x)
