import time

import tensorflow as tf


class BaseTrain(object):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.valid_data = valid_data
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def train(self):
        print("[*] Start training, with start epoch %d start iter %d: " % (
            self.model.cur_epoch_tensor.eval(self.sess), self.model.global_step_tensor.eval(self.sess)))
        self.start_time = time.time()
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        print("[*] Finish training.")

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
