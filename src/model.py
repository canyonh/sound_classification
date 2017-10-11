import tensorflow as tf


class LinearModel:

    def __init__(self, dimension, num_classes, learning_rate):
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

    # @todo 
    def Train(training_set, epoch_cnt, batch, batch_size=500):
        with tf.Session() as sess:
            num_train = training_set
            tf.global_variables_initializer().run()

            # training
            for epoch_idx in range(epoch_cnt):
                epoch_loss = 0
                current = 0
