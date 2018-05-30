from __future__ import division

import tensorflow as tf

from base.base_model import BaseModel


class DenseNet(BaseModel):
    def __init__(self, config):
        super(DenseNet, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.input_placeholder()
        self.build_graph()

    def input_placeholder(self):
        self.is_training = tf.placeholder(tf.bool)

        self.images = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape, name='input_image')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def build_graph(self):
        self.mrsa_init = tf.contrib.layers.variance_scaling_initializer()
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.name_scope("conv1"):
            output = tf.layers.conv2d(self.images, filters=16, kernel_size=3, strides=1, padding="same",
                                      kernel_initializer=self.mrsa_init)

        for block in range(self.config.n_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.denseBlock(output, self.config.growth_rate, self.config.layers_per_block)
            if block != self.config.n_blocks - 1:
                with tf.variable_scope("Transform_%d" % block):
                    output = self.transformLayer(output)

        with tf.variable_scope("Output"):
            self.logits = self.outLayer(output)

        self.prediction = tf.nn.softmax(self.logits)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = self.cross_entropy + self.l2_loss * self.config.weight_decay

        # optimizer = tf.train.AdamOptimizer(self.learning_rate, name="AdamOptimizer")
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_step = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate, name="Adam").minimize(self.loss,
                                                                                           global_step=self.global_step_tensor)

        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.prediction

    def denseBlock(self, x, growth_rate, layers_per_block):
        output = x
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.unitLayer(output, growth_rate)

        return output

    def unitLayer(self, x, growth_rate):
        # output = tf.nn.relu(tf.layers.batch_normalization(x, training=self.is_training))
        output = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        output = tf.layers.conv2d(output, filters=growth_rate, kernel_size=3, padding="same",
                                  kernel_initializer=self.mrsa_init)
        output = tf.layers.dropout(output, rate=self.config.dropout_rate, training=self.is_training)
        output = tf.concat([output, x], axis=3)

        return output

    def transformLayer(self, x):
        out_features = int(int(x.get_shape()[-1]) * self.config.reduction)
        # output = tf.nn.relu(tf.layers.batch_normalization(x, training=self.is_training))
        output = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        output = tf.layers.conv2d(output, filters=out_features, kernel_size=1, padding="valid",
                                  kernel_initializer=self.mrsa_init)
        output = tf.layers.dropout(output, rate=self.config.dropout_rate, training=self.is_training)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

        return output

    def outLayer(self, x):
        # output = tf.nn.relu(tf.layers.batch_normalization(x, training=self.is_training))
        output = tf.nn.relu(self.batch_normalization(x, training=self.is_training))

        feature_size = int(output.get_shape()[-2])
        output = tf.layers.average_pooling2d(output, pool_size=feature_size, strides=1)

        features_num = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_num])
        logits = tf.layers.dense(output, units=self.config.n_classes, kernel_initializer=self.xavier_init)

        return logits

    def batch_normalization(self, x, training):
        depth = x.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed_tensor

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
