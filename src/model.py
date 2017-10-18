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
    with LinearModel() as linear_model:
        linear_model.Train(test_dataset, 1000, 1)
        infer_result = linear_model.Infer(test_dataset.x[0])
        for result in infer_result:
            logging.info("infer result: %s", str(result))


class LinearModel:

    # move learning rate to train
    # in train, get parameters from training_set and set it
    # in infer we could check it
    def __init__(self):
        self.session = None
        return

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        if self.session is not None:
            self.session.close()

    # @todo
    def Train(self, training_set, epoch_cnt,
              batch_size=500, learning_rate=5e-4):
        logging.info("x shape = %s, y shape = %s",
                     str(training_set.x.shape),
                     str(training_set.y.shape))
        num_samples = training_set.x.shape[0]
        assert num_samples == training_set.y.shape[0]
        dimension = training_set.x.shape[1]
        num_classes = training_set.y.shape[1]
        logging.info("Train: sample_cnt: %d, dim: %d, classes: %d",
                     num_samples, dimension, num_classes)

        # define graph
        self.learning_rate = learning_rate
        self.dimension = dimension
        self.num_classes = num_classes

        self.x = tf.placeholder(tf.float32, [None, dimension])
        self.y_correct = tf.placeholder(tf.float32, [None, num_classes])
        self.W = tf.Variable(tf.random_normal([dimension, num_classes]))
        self.b = tf.Variable(tf.random_normal([num_classes]))
        self.y_output = tf.matmul(self.x, self.W) + self.b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_correct, logits=self.y_output)
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )

        self.accuracy = 0.0
        if self.session is not None:
            self.session.close()

        self.session = tf.Session()
        # @todo session lifetime management
        num_train = len(training_set.x)
        self.session.run(tf.global_variables_initializer())

        # training
        for epoch in range(epoch_cnt):
            epoch_loss = 0
            current = 0
            for _ in range(int(num_train/batch_size)):
                c, _ = self.session.run([self.cost, self.optimizer],
                                        feed_dict={self.x: training_set.x,
                                        self.y_correct: training_set.y})
                epoch_loss += c
            current += batch_size
            logging.info("Epoch: %d, loss: %f", epoch, epoch_loss)
        correct_prediction = tf.equal(tf.argmax(self.y_output, 1),
                                      tf.argmax(self.y_correct, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                       tf.float32))
        logging.info("Accuracy: %s", str(self.accuracy))

    def Infer(self, x):
        result = self.session.run([self.y_output], feed_dict={self.x: x})
        return tf.argmax(result, 1)
