import tensorflow as tf


class LinearModel:

    def __init__(self, dimension, num_classes):
        self.dimension = dimension
        self.num_classes = num_classes
        self.X = tf.placeholder(tf.float32, [None, dimension])
        self.y_correct = tf.placeholder(tf.float32, [None, num_classes])
        self.W = tf.Variable(tf.random_normal([dimension, num_classes]))
        self.b = tf.variable(tf.random_normal([num_classes]))
        self.y_output = tf.matmul(self.X, self.W) + self.b
