import logging
import tensorflow as tf
import numpy as np


#
# Simple Models - Linear and NN
#
class ModelParams:
    learning_rate = 0.0
    num_classes = 0
    dimension = 0


class ModelGraphs:
    x = None
    y_correct = None
    optimizer = None
    y_output = None
    correct_prediction = None
    accuracy_node = None
    W = None
    b = None
    y_output = None
    cost = None


class SimpleModel:
    saver = None
    training_accuracy = None
    session = None
    params = ModelParams()
    graph = ModelGraphs()

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        if self.session is not None:
            self.session.close()

    def DefineModelImpl(self):
        if self.session is not None:
            raise TypeError
        raise NotImplementedError

    def Train(self, training_set, epoch_cnt,
              batch_size=500, learning_rate=5e-4):

        if self.session is not None:
            raise TypeError

        # calculate num_classes; dimension... etc
        logging.info("x shape = %s, y shape = %s, batch size: %d",
                     str(training_set.x.shape),
                     str(training_set.y.shape),
                     batch_size)
        num_samples = training_set.x.shape[0]
        assert num_samples == training_set.y.shape[0]
        self.params.dimension = training_set.x.shape[1]
        self.params.num_classes = training_set.y.shape[1]
        self.params.learning_rate = learning_rate
        logging.info("Train: sample_cnt: %d, dim: %d, classes: %d",
                     num_samples,
                     self.params.dimension,
                     self.params.num_classes)

        # define data (self.graph.x) and correct label (self.graph.y_correct)
        self.graph.x = \
            tf.placeholder(tf.float32, [None, self.params.dimension])
        self.graph.y_correct = \
            tf.placeholder(tf.float32, [None, self.params.num_classes])

        # define the rest of the model using thru DefineModelImpl
        self.DefineModelImpl()

        # check accuracy
        self.graph.correct_prediction = \
            tf.equal(tf.argmax(self.graph.y_output, 1),
                     tf.argmax(self.graph.y_correct, 1))

        self.graph.accuracy_node = \
            tf.reduce_mean(tf.cast(self.graph.correct_prediction,
                                   tf.float32))

        # open a new session
        if self.session is not None:
            self.session.close()

        self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        # perform the training
        self.TrainImpl(training_set, epoch_cnt, batch_size)

    def TrainImpl(self, training_set, epoch_cnt, batch_size):

        num_train = len(training_set.x)

        # training
        for epoch in range(epoch_cnt):
            epoch_loss = 0
            current = 0
            for _ in range(int(num_train/batch_size)):
                fetch = [self.graph.cost, self.graph.optimizer]
                feed = {self.graph.x:
                        training_set.x[current:current + batch_size],
                        self.graph.y_correct:
                        training_set.y[current:current + batch_size]}
                c, _ = self.session.run(fetch, feed)
                epoch_loss += c
                current += batch_size
            logging.info("Epoch: %d, loss: %f", epoch, epoch_loss)

        self.training_accuracy = \
            self.session.run(self.graph.accuracy_node,
                             feed_dict={self.graph.x:
                                        training_set.x,
                                        self.graph.y_correct:
                                        training_set.y})

        logging.info("accuracy: %f", self.training_accuracy)

    def Infer(self, x):
        if self.session is None:
            raise TypeError

        result = self.session.run([self.graph.y_output],
                                  feed_dict={self.graph.x: x})
        scores = np.reshape(result[0], result[0].shape[1])
        logging.debug("scores: %s", str(scores))
        correct_class = np.argmax(scores)
        logging.debug("inferred class: %d", correct_class)
        return correct_class

    def SaveModel(self, save_path):
        if self.session is None:
            raise TypeError
        saved = self.saver.save(self.session, save_path, global_step=1000)
        logging.info("Mode saved in file: %s" % saved)

    def LoadModel(self, saved_path):
        if self.session is not None:
            raise TypeError

        self.session = tf.Session()
        saver = tf.train.import_meta_graph('/tmp/nn-1000.meta')
        saver.restore(self.session, tf.train.latest_checkpoint('/tmp'))

        logging.info("Model restored: %s", saved_path)


class SimpleLinearModel(SimpleModel):
    def DefineModelImpl(self):
        self.graph.W = \
            tf.Variable("W", tf.random_normal([self.params.dimension,
                        self.params.num_classes]))

        self.graph.b = \
            tf.Variable("b", tf.random_normal([self.params.num_classes]))

        self.graph.y_output = \
            tf.matmul(self.graph.x, self.graph.W) + self.graph.b

        self.graph.cost = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                           labels=self.graph.y_correct,
                           logits=self.graph.y_output))

        self.graph.optimizer = \
            tf.train.AdamOptimizer(self.params.learning_rate) \
            .minimize(self.graph.cost)

        self.graph.correct_prediction = \
            tf.equal(tf.argmax(self.graph.y_output, 1),
                     tf.argmax(self.graph.y_correct, 1))

        self.graph.accuracy_node = \
            tf.reduce_mean(tf.cast(self.graph.correct_prediction,
                                   tf.float32))


class SimpleNeuralNetwork(SimpleModel):
    def __init__(self, num_layers, neurons_per_layer):
        self.params.num_layers = num_layers
        self.params.neurons_per_layer = neurons_per_layer
        return

    def DefineModelImpl(self):
        logging.info("DefineModel() dim: %d, classes: %d",
                     self.params.dimension, self.params.num_classes)

        self.graph.W = \
            tf.Variable(tf.random_normal([self.params.dimension,
                                          self.params.num_classes]))

        self.graph.b = \
            tf.Variable(tf.random_normal([self.params.num_classes]))

        self.graph.y_output = \
            tf.matmul(self.graph.x, self.graph.W) + self.graph.b

        self.graph.cost = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                           labels=self.graph.y_correct,
                           logits=self.graph.y_output))

        self.graph.optimizer = \
            tf.train.AdamOptimizer(self.params.learning_rate) \
            .minimize(self.graph.cost)

        self.graph.correct_prediction = \
            tf.equal(tf.argmax(self.graph.y_output, 1),
                     tf.argmax(self.graph.y_correct, 1))
        self.graph.accuracy_node = \
            tf.reduce_mean(tf.cast(self.graph.correct_prediction,
                                   tf.float32))


# @todo re-implemented using simple model. delete if no longer needed
class LinearModel:
    session = None
    learning_rate = None
    dimension = None
    num_classes = None
    training_accuracy = 0.0
    tf_x = None
    tf_y_correct = None
    tf_W = None
    tf_b = None
    tf_y_output = None
    tf_cost = None
    tf_optimizer = None
    tf_correct_prediction = None
    tf_accuracy = None

    # move learning rate to train
    # in train, get parameters from training_set and set it
    # in infer we could check it
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
        self.learning_rate = learning_rate
        self.dimension = training_set.x.shape[1]
        self.num_classes = training_set.y.shape[1]
        logging.info("Train: sample_cnt: %d, dim: %d, classes: %d",
                     num_samples, self.dimension, self.num_classes)

        # define graph
        self.tf_x = tf.placeholder(tf.float32, [None, self.dimension])
        self.tf_y_correct = tf.placeholder(tf.float32,
                                           [None, self.num_classes])
        self.tf_W = tf.Variable(tf.random_normal([self.dimension,
                                                 self.num_classes]))
        self.tf_b = tf.Variable(tf.random_normal([self.num_classes]))
        self.tf_y_output = tf.matmul(self.tf_x, self.tf_W) + self.tf_b
        self.tf_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.tf_y_correct, logits=self.tf_y_output)
        )
        self.tf_optimizer = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.tf_cost)

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
                feed = {self.tf_x:
                        training_set.x[current:current + batch_size],
                        self.tf_y_correct:
                        training_set.y[current:current + batch_size]}
                c, _ = self.session.run([self.tf_cost, self.tf_optimizer],
                                        feed)
                epoch_loss += c
                current += batch_size

            logging.info("Epoch: %d, loss: %f", epoch, epoch_loss)

        # check accuracy
        self.tf_correct_prediction = tf.equal(tf.argmax(self.tf_y_output, 1),
                                              tf.argmax(self.tf_y_correct, 1))
        self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_correct_prediction,
                                          tf.float32))

        self.training_accuracy = self.session.run(
                                         self.tf_accuracy,
                                         feed_dict={self.tf_x:
                                                    training_set.x,
                                                    self.tf_y_correct:
                                                    training_set.y})
        logging.info("accuracy: %f", self.training_accuracy)

    def Infer(self, x):
        result = self.session.run([self.tf_y_output], feed_dict={self.tf_x: x})
        scores = np.reshape(result[0], result[0].shape[1])
        logging.debug("scores: %s", str(scores))
        correct_class = np.argmax(scores)
        logging.debug("inferred class: %d", correct_class)
        return correct_class
