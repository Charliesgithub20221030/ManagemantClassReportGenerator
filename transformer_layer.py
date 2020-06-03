import math
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

# pylint: disable=invalid-name

_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


class Projection (object):
    """ dense and reshape """

    def __init__(self, hidden_size, _num_heads):
        self._dense_layer = tf.layers.Dense(hidden_size, use_bias=False)
        self._num_heads = _num_heads
        self._hidden_size = hidden_size

    def __call__(self, x):
        x_shape = tf.shape(x)
        if self._num_heads is not None:
            x = self._dense_layer(x)
            x = tf.reshape(x, [x_shape[0],  x_shape[1], self._num_heads, -1])

        x = tf.transpose(x, [0, 2, 1, 3])
        if self._num_heads is None:
            x = tf.reshape(x, [x_shape[0], -1, self._hidden_size])
            x = self._dense_layer(x)

        return x


class Attention(object):
    def __init__(self, hidden_size, num_heads, attention_dropout):
        self.k_projection = Projection(
            hidden_size, num_heads
        )
        self.q_projection = Projection(
            hidden_size, num_heads
        )
        self.v_projection = Projection(
            hidden_size, num_heads
        )

        self.output_projection = Projection(
            hidden_size
        )
        self.attention_dropout = attention_dropout
        self.scale = (hidden_size // num_heads) ** -.5

    def __call__(self, x, y, bias, training):
        q = self.q_projection(x) * self.scale
        k = self.k_projection(y)
        v = self.v_projection(y)

        logits = tf.matmul(q, k, transpose_b=True) + bias

        output = tf.nn.softmax(logits)
        output_shape = tf.shape(output)
        output = tf.layers.dropput(
            output,
            rate=self.attention_dropout,
            noise_shape=[1, 1, output_shape[2], output_shape[3]],
            training=training
        )
        output = tf.matmul(output, v)

        return self.output_projection(output)


class SelfAttention(Attention):
    def __call__(self, x, bias, training):
        return super(SelfAttention, self).__call__(x, x, bias, training)


class FeedForward(object):
    def __init__(self, output_size, filter_size, dropout):
        self._relu_layer = tf.layers.Dense(
            filter_size, activation=tf.nn.relu)
        self._output_layer = tf.layers.Dense(output_size)
        self._dropout = dropout

    def __call__(self, input, training):
        with tf.variable_scope("ffn"):
            x = self._relu_layer(input)
            x_shape = tf.shape(x)
            x = tf.layers.dropout(
                x,
                self._dropout,
                noise_shape=[x_shape[0], x_shape[2]],
                training=training
            )
            x = self._output_layer(x)

        return x


class PreProcess(object):
    def __call__(self, input):
        x = tf.layers.LayerNormalization(input, axis=2)
        return x


class PostProcess(object):
    def __init__(self, dropout):
        self.dropout = dropout

    def __call__(self, x, y, training):
        y_shape = tf.shape(y)
        y = tf.layers.Dropout(
            y,
            self.dropout,
            noise_shape=[y_shape[0], 1, y_shape[2]],
            training=training
        )
        x += y
        return x


class Encoder(object):
    def __init__(self, hidden_size, filter_size, num_heads, attention_dropout, relu_dropout, postprocess_dropout):
        self.selfattn_preprocess = PreProcess()
        self.selfattn_layer = SelfAttention(
            hidden_size, num_heads, attention_dropout)
        self.selfattn_postprocess = PostProcess(postprocess_dropout)

        self.ffn_preprocess = PreProcess()
        self.ffn_layer = FeedForward(hidden_size,  filter_size, relu_dropout)
        self.ffn_postprocess = PostProcess(postprocess_dropout)

    def __call__(self, input, bias, training):
        x = input
        with tf.variable_scope("self_attention"):
            y = self.selfattn_layer(self.attn_preprocess(x), bias, training)
            x = self.seflattn_postprocess(x, y, training)
        with tf.variable_scope('ffn'):
            y = self.ffn_layer(self.ffn_preprocess(x), training)
            x = self.ffn_postprocess(x, y, training)
        return x


class Decoder(object):
    def __init__(self,  hidden_size, filter_size,  num_heads, attention_dropout, relu_dropout, postprocess_dropout):
        self.selfattn_preprocess = PreProcess()
        self.selfattn_layer = SelfAttention(
            hidden_size, num_heads, attention_dropout)
        self.selfattn_postprocess = PostProcess(postprocess_dropout)

        self.attn_preprocess = PreProcess()
        self.attn_layer = Attention(hidden_size, num_heads, attention_dropout)
        self.attn_postprocess = PostProcess(postprocess_dropout)

        self.ffn_preprocess = PreProcess()
        self.ffn_layer = FeedForward(hidden_size, filter_size, relu_dropout)
        self.ffn_postprocess = PostProcess(postprocess_dropout)

    def __call__(self, x, decoder_self_attention_bias, input, input_attention_bias, training):
        with tf.variable_scope("self_attention"):
            y = self.selfattn_layer(
                self.selfattn_preprocess(
                    x), decoder_self_attention_bias, training
            )
            x = self.selfattn_postprocess(x, y, training)

        with tf.variable_scope('decoder_attention'):
            y = self.attn_layer(
                self.attn_preprocess(x), training
            )
            x += self.ffn_postprocess(x, y, training)
        with tf.variable_scope('ffn'):
            y = self.ffn_layer(
                self.ffn_preprocess(x), training
            )
            x += self.ffn_postprocess(x, y, training)
        return x


class TimingSignal(object):
    """Layer that adds a transformer-style timing signal to inputs.
    Equivalent to tensor2tensor's 1d timing signal, generalized to allow each
    example in a batch to begin at a different index. See
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    def __call__(self, inputs, start_index=None):
        dtype = inputs.dtype
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        length = inputs_shape[1]
        channels = inputs_shape[2]
        if start_index is None:
            start_index = tf.zeros((batch_size, 1), tf.int32)

        position = tf.expand_dims(tf.range(length), 0)
        position = tf.tile(position, [batch_size, 1]) + start_index
        position = tf.cast(position, dtype)
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(_MAX_TIMESCALE / _MIN_TIMESCALE) /
            tf.maximum(tf.cast(num_timescales, dtype) - 1, 1))
        inv_timescales = _MIN_TIMESCALE * tf.exp(
            tf.cast(tf.range(num_timescales), dtype) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 2) * tf.reshape(
            inv_timescales, [1, 1, -1])
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [-1, length, channels])
        return inputs + signal


class TransformerEncoder(object):
    def __init__(self, hidden_size, filter_size, num_layers, num_heads, attention_dropout, relu_dropout, postprocess_dropout):
        self.postprocess_dropout = postprocess_dropout
        self.preprosess_layer = PreProcess()

        self.encoder_layer = [
            Encoder(hidden_size, filter_size, num_heads, attention_dropout,
                    relu_dropout, postprocess_dropout) for _ in range(num_layers)
        ]

    def __call__(self, input_BIH, padding_BI,  training, cache=None):
        if cache is not None and "encoder_output" in cache:
            return cache['encoder_output']

        attention_biasB11I = tf.expand_dims(
            tf.expand_dims(padding_BI * padding_BI.dtype.min, 1), 1
        )
        encoder_input_BIH = TimingSignal()(input_BIH)
        encoder_input_BIH = tf.layers.dropout(
            encoder_input_BIH, self.postprocess_dropout, training
        )

        x_BIH = encoder_input_BIH
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            for i, encoder_layer in enumerate(self.encoder_layer):
                with tf.variable_scope('layer_%d' % i):
                    x_BIH = encoder_layer(x_BIH, attention_biasB11I, training)
            encoder_output_BIH = self.preprocess_layer(x_BIH)

        if cache is not None:
            cache['encoder_output'] = encoder_output_BIH


class TransformerDecoder(object):
    def __init__(self, hidden_size, filter_size, num_heads, attention_dropout, relu_dropout, postprocess_dropout):
        self.postprocess_dropout = postprocess_dropout
        self.preprocess_layer = PreProcess()

        self.decoder_layer = [
            Decoder(hidden_size, filter_size, num_heads, attention_dropout, relu_dropout, postprocess_dropout) for _ in range(num_layers)
        ]

    def __call__(self, input_BIH, input_padding_BI, targets_BTH, training, targets_start=None):
        inputs_attention_bias_B11I = tf.expand_dims(
            tf.expand_dims(input_padding_BI *
                           input_padding_BI .dtype.min, 1), 1
        )
        targets_len = tf.shape(targets_BTH)[1]
        upper_triangular_TT = 1-tf.matrix_band_part(
            tf.ones((targets_len, targets_len), dtype=input_BIH.dtype), -1, 0
        )
    # matrix_band_part (input , i , j): 左邊第 i 個起算到右下角的對角線保留，上面第 j 個起算的對角線也保留，剩下都為 0
    # 所以 (input , -1 , 0) 表示又上半全砍 -> upper triangular matrix
        decoder_self_attention_bias_11TT = tf.expand_dims(
            tf.expand_dims(upper_triangular_TT, 0), 0) * input_BIH.dtype.min

        decoder_input_BTH = tf.pad(
            targets_BTH, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input_BTH = TimingSignal()(decoder_input_BTH, targets_start)
        decoder_input_BTH = tf.layers.dropout(
            decoder_input_BTH, self.postprocess_dropout, training=training
        )

        x_BTH = decoder_input_BTH

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            for i, decoder_layer in enumerate(self.decoder_layer):
                with tf.variable_scope("layer_%d" % i):
                    x_BTH = decoder_layer(
                        x_BTH,
                        decoder_self_attention_bias_11TT,
                        input_BIH, inputs_attention_bias_B11I, training
                    )
            decoder_output_BTH = self.preprocess_layer(x_BTH)
        return decoder_output_BTH


class TransformerEncDec(object):
    def __init__(self, hidden_size, filter_size, num_encoder_layers, num_decoder_layers,
                 num_enc_heads, num_dec_heads, attention_dropout, relu_dropout, postprocess_dropout):
        self.encoder_layer = TransformerEncoder(
            hidden_size, filter_size, num_encoder_layers, num_enc_heads, attention_dropout, relu_dropout, postprocess_dropout)
        self.decoder_layer = TransformerDecoder(
            hidden_size, filter_size, num_decoder_layers, num_dec_heads, attention_dropout, relu_dropout, postprocess_dropout)

    def __call__(
        self, inputs_BIH, targets_BTH, padding_BI, training, cache=None, targets_start=None
        ):

        encoder_output_BIH = self.encoder_layer(
            input_BIH, padding_BI, training, cache=cache)
        decoder_output_BTH = self.decoder_layer(
            encoder_output_BIH, padding_BI, targets_BTH, training, targets_start=tartgets_start)
        )
        return decoder_output_BTH
