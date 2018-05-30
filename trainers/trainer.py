from __future__ import division

import time
import csv
import numpy as np

from base.base_train import BaseTrain


class DenseNetTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(DenseNetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)
        self.learning_rate = self.config.learning_rate
        self.train_samples = len(self.train_data.dataset)
        self.num_iter_per_epoch = self.train_samples // self.config.batch_size

    def train_epoch(self):
        losses = []
        accs = []
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        if cur_epoch == self.config.reduce_lr_epoch_1 or cur_epoch == self.config.reduce_lr_epoch_2:
            self.learning_rate /= 10
        if cur_epoch % self.config.valid_num_epoch == 0:
            self.evaluate()
        for itr in range(self.num_iter_per_epoch):
            loss, acc = self.train_step()

            print("Epoch: [%2d] [%4d/%4d] time: % 4.4f, loss : %.6f" % (
                cur_epoch + 1, itr + 1, self.num_iter_per_epoch, time.time() - self.start_time, loss))

            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {'loss': loss, 'acc': acc}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        self.learning_rate = self.learning_rate / 10
        batch_images, batch_labels = self.train_data.next_batch(self.config.batch_size)
        feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels,
                     self.model.learning_rate: self.learning_rate, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc

    def evaluate(self):
        valid_samples = len(self.valid_data.dataset)
        losses = []
        accs = []
        for i in range(valid_samples // self.config.batch_size):
            batch_images, batch_labels = self.valid_data.next_batch(self.config.batch_size)
            feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels,
                         self.model.is_training: False}
            loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        print("--- Test --- Average loss %.2f, accuracy %.2f ---" % (loss, acc))

        summaries_dict = {'loss': loss, 'acc': acc}
        self.logger.summarize(self.model.global_step_tensor.eval(self.sess), scope="test",
                              summaries_dict=summaries_dict)

    def test(self, output_prediction_path):
        csvfile = open(output_prediction_path, 'w')
        writer = csv.writer(csvfile)
        valid_samples = len(self.valid_data.dataset)
        _, _, last_name = self.valid_data.get_single_sample(0)
        count = 0
        prediction_sum = 0
        for index in range(valid_samples):
            image, label, name = self.valid_data.get_single_sample(index)
            feed_dict = {self.model.images: image, self.model.labels: label,
                         self.model.is_training: False}
            pred = self.sess.run([self.model.prediction], feed_dict=feed_dict)
            prediction = np.argmax(pred)
            if name[:name.rfind('/') + 1] != last_name[:last_name.rfind('/') + 1]:
                if ((2 * prediction_sum) > count):
                    writer.writerow([last_name[:last_name.rfind('/') + 1], int(1)])
                else:
                    writer.writerow([last_name[:last_name.rfind('/') + 1], int(0)])
                count = 1
                prediction_sum = prediction
            elif name[:name.rfind('/') + 1] == last_name[:last_name.rfind('/') + 1]:
                count += 1
                prediction_sum += prediction
            last_name = name
        if ((2 * prediction_sum) > count):
            writer.writerow([last_name[:last_name.rfind('/') + 1], int(1)])
        else:
            writer.writerow([last_name[:last_name.rfind('/') + 1], int(0)])

            # writer.writerow([name[:name.rfind('/')+1], prediction])
        csvfile.close()
