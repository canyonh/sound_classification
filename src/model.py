import os
import logging
import dataset
import common
import tensorflow as tf
import numpy as np


def TestLinearModel():
    test_dataset = dataset.NpyDataSet()
    dir_path = os.path.join(common.DataDir(), "systest-prototype-small-npy")
    print(dir_path)
    test_dataset.Load(dir_path)
    with LinearModel() as linear_model:
        linear_model.Train(test_dataset, 10, 1)
        first_row = test_dataset.x[0]
        first_row = first_row.reshape(1, first_row.shape[0])
        infer_result = linear_model.Infer(first_row)
        logging.info("infer result: %d", infer_result)


class ModelParams:
    def __init__(self):
        self.accuracy = 0.0
        self.learning_rate = 0.0


class ModelGraphs:
    def __init__(self):
        self.x = None
        self.y_correct = None
        self.optimizer = None
        self.y_output = None


class SimpleModel:

    def __init__(self):
        self.session = None
        self.params = ModelParams()
        self.graph = ModelGraphs()
        return

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        if self.session is not None:
            self.session.close()

    def DefineModelImpl(self, dimension, num_classes):
        if self.session is not None:
            raise TypeError
        raise NotImplementedError

    def Test(self):
        return

    def Train(self, training_set, epoch_cnt,
              batch_size=500, learning_rate=5e-4):

        if self.session is not None:
            raise TypeError

        logging.info("x shape = %s, y shape = %s",
                     str(training_set.x.shape),
                     str(training_set.y.shape))
        num_samples = training_set.x.shape[0]
        assert num_samples == training_set.y.shape[0]
        dimension = training_set.x.shape[1]
        num_classes = training_set.y.shape[1]
        logging.info("Train: sample_cnt: %d, dim: %d, classes: %d",
                     num_samples, dimension, num_classes)

        self.graph.x = tf.placeholder(tf.float32, [None, dimension])
        self.graph.y_correct = tf.placeholder(tf.float32, [None, num_classes])

        SimpleModel.DefineModelImpl(self, dimension, num_classes)

    def Infer(self, x):
        if self.session is None:
            raise TypeError
        raise NotImplementedError

    def SaveModel(self, model_file):
        if self.session is None:
            raise TypeError
        raise NotImplementedError


class SimpleLinearModel(SimpleModel):
    def DefineModelImpl(self, dimension, num_classes):
        self.graph.W = tf.Variable(tf.random_normal([dimension, num_classes]))
        self.graph.b = tf.Variable(tf.random_normal([num_classes]))
        self.graph.y_output = tf.matmul(self.tf_x, self.tf_W) + self.tf_b
        self.graph.cost = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                           labels=self.graph.y_correct,
                           logits=self.graph.y_output))
        self.graph.optimizer = \
            tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.graph.cost)

        self.tf_correct_prediction = tf.equal(tf.argmax(self.tf_y_output, 1),
                                              tf.argmax(self.tf_y_correct, 1))
        self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_correct_prediction,
                                          tf.float32))


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

        self.tf_x = tf.placeholder(tf.float32, [None, dimension])
        self.tf_y_correct = tf.placeholder(tf.float32, [None, num_classes])
        self.tf_W = tf.Variable(tf.random_normal([dimension, num_classes]))
        self.tf_b = tf.Variable(tf.random_normal([num_classes]))
        self.tf_y_output = tf.matmul(self.tf_x, self.tf_W) + self.tf_b
        self.tf_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.tf_y_correct, logits=self.tf_y_output)
        )
        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.tf_cost
        )

        if self.session is not None:
            self.session.close()

        self.session = tf.Session()
        num_train = len(training_set.x)
        self.session.run(tf.global_variables_initializer())

        # training
        for epoch in range(epoch_cnt):
            epoch_loss = 0
            current = 0
            for _ in range(int(num_train/batch_size)):
                c, _ = self.session.run([self.tf_cost, self.tf_optimizer],
                                        feed_dict={self.tf_x: training_set.x,
                                        self.tf_y_correct: training_set.y})
                epoch_loss += c
            current += batch_size
            logging.info("Epoch: %d, loss: %f", epoch, epoch_loss)

        # check accuracy
        self.tf_correct_prediction = tf.equal(tf.argmax(self.tf_y_output, 1),
                                              tf.argmax(self.tf_y_correct, 1))
        self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_correct_prediction,
                                          tf.float32))

        self.accuracy = self.session.run(self.tf_accuracy,
                                         feed_dict={self.tf_x:
                                                    training_set.x,
                                                    self.tf_y_correct:
                                                    training_set.y})
        logging.info("accuracy: %f", self.accuracy)

    def Infer(self, x):
        result = self.session.run([self.tf_y_output], feed_dict={self.tf_x: x})
        scores = np.reshape(result[0], result[0].shape[1])
        logging.debug("scores: %s", str(scores))
        correct_class = np.argmax(scores)
        logging.debug("inferred class: %d", correct_class)
        return correct_class
