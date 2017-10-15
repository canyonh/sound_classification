import os
import logging
import dataset
import common
import tensorflow as tf


def TestLinearModel():
    test_dataset = dataset.NpyDataSet()
    dir_path = os.path.join(common.DataDir(), "systest-prototype-small-npy")
    print(dir_path)
    test_dataset.Load(dir_path)
    linear_model = LinearModel()
    linear_model.Train(test_dataset, 10, 1)


class LinearModel:

    # move learning rate to train
    # in train, get parameters from training_set and set it
    # in infer we could check it
    def __init__(self):
        return

    # @todo
    def Train(self, training_set, epoch_cnt,
              learning_rate=5e-4, batch_size=500):
        dimension = training_set.x.shape[1]
        num_classes = training_set.y.shape[1]

        assert training_set.x.shape[1] == dimension
        assert training_set.y.shape[0] == dimension

        # define graph
        self.learning_rate = learning_rate
        self.dimension = dimension
        self.num_classes = num_classes

        self.X = tf.placeholder(tf.float32, [None, dimension])
        self.y_correct = tf.placeholder(tf.float32, [None, num_classes])
        self.W = tf.Variable(tf.random_normal([dimension, num_classes]))
        self.b = tf.variable(tf.random_normal([num_classes]))
        self.y_output = tf.matmul(self.X, self.W) + self.b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_, logits=self.y_output)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        self.accuracy = 0.0
		# @todo session lifetime management
        with tf.Session() as sess:
            num_train = len(training_set.x)
            tf.global_variables_initializer().run()

            # training
            for epoch in range(epoch_cnt):
                epoch_loss = 0
                current = 0
                for _ in range(int(num_train/batch_size)):
                    c, _ = sess.run([self.cost, self.optimizer],
                                    feed_dict={self.x: training_set.x,
                                    self.y_correct: training_set.y})
                epoch_loss += c
                current += batch_size
                logging.info("Epoch: %d, loss: %f", epoch, epoch_loss)
            correct_prediction = tf.equal(tf.arg_max(self.y_output),
                                          tf.argmax(self.y_correct, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                           tf.float32))
            logging.info("Accuracy: %f", self.accuracy)

    def Infer(self):
        raise NotImplementedError
