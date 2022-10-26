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
            x = Dense(64, activation='relu')(x)
        return Dense(out_size, activation=activation)(x)

    def get_main_model(self, x):
        # encoder
        h = self.encoder(x)

        # regularization
        l = self.regularization(h)

        # decoder
        y1, y2 = self.decoder(l)
        return [y1, y2], h

    # baseline ---------------------------------------
    def baseline_encoder(self, x):
        x = tf.keras.layers.Embedding(2, 8)(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.ff(2, x, 'linear', 2)
        return x

    def baseline_regularization(self, l):
        return l

    def baseline_decoder(self, x):
        x = self.ff(16, x, 'relu', 2)
        y1 = self.get_output_layer(x, 'y1')
        y2 = self.get_output_layer(x, 'y2')
        return y1, y2

    # proposed ---------------------------------------
    def get_a_hidden_node(self, x1, r):
        A = self.ff(2, x1, 'softmax', depth=8)
        A = tf.expand_dims(A, 1)
        a = tf.matmul(A, r)
        a = tf.reshape(a, [-1, 1])
        a = self.ff(1, a, 'linear')
        return a

    def proposed_encoder(self, x):
        x1, xr = tf.split(x, [1, 2], -1)
        r = tf.keras.layers.Embedding(2, 1)(xr)
        l1 = self.get_a_hidden_node(x1, r)
        l2 = self.get_a_hidden_node(x1, r)
        l = tf.concat([l1, l2], -1)
        return l

    def proposed_regularization(self, l):
        l = tf.keras.layers.ActivityRegularization(l2=self.args.beta)(l)
        l = tf.keras.layers.GaussianNoise(self.args.alpha)(l)
        return l

    def proposed_decoder(self, x):
        l1, l2 = tf.split(x, [1, 1], -1)
        b = self.ff(2, l1, 'softmax')
        c = self.ff(2, l2, 'softmax')
        b = tf.expand_dims(b, -1)
        c = tf.expand_dims(c, 1)
        s = tf.matmul(b, c)
        s = tf.keras.layers.Flatten()(s)
        s = tf.split(s, [1, 1, 1, 1], -1)
        s[2], s[3] = s[3], s[2]
        y1 = tf.concat([s[0] + s[1], s[2] + s[3]], -1)
        y2 = tf.concat([s[0] + s[2], s[1] + s[3]], -1)
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
