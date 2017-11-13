import os
import glob
import numpy as np
import tensorflow as tf
import common
import librosa
import librosa.display
import matplotlib.pyplot as plt
import logging
import time
from tensorflow.examples.tutorials.mnist import input_data

#
# Sample preparation
#

sampling_rate = 8000
num_fft = 320
hop_len = 160
num_mels = 160
duration = 3.0


def PlotSpectrum(spectro, title="no titie"):
    log_specto = librosa.core.logamplitude(spectro)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    librosa.display.specshow(log_specto,
                             sr=sampling_rate,
                             x_axis='time', y_axis='mel',
                             hop_length=hop_len)
    plt.show()


def ConvertSample(src_filename, target_filename, plot_spectrum=False):
    y, _ = librosa.core.load(src_filename, sr=sampling_rate, duration=duration)
    logging.debug(y.shape)
    spectro = librosa.feature.melspectrogram(y, sr=sampling_rate,
                                             n_fft=num_fft,
                                             hop_length=hop_len,
                                             n_mels=num_mels)
    logging.debug(spectro.shape)
    if plot_spectrum:
        PlotSpectrum(spectro, title=target_filename)

    np.save(target_filename, spectro)

    # load test
    # spectro_loaded = np.load(target_filename + '.npy')
    # PlotSpectrum(spectro_loaded)


def LoadSpectro(filename, plot_spectrum=True):
    y = np.load(filename)
    PlotSpectrum(y)


def PrepareInput(src_dir, target_dir):
    start = time.time()
    # cleanup current target dir
    if os.path.exists(target_dir):
        assert os.path.isdir(target_dir)
        old_files = glob.glob(os.path.join(target_dir, "*"))
        for f in old_files:
            os.remove(f)
    else:
        os.mkdir(target_dir)

    # load files and convert to melspectrum
    files = glob.glob(os.path.join(src_dir, "*.wav"))

    show_first_spectrogram = False
    sample_converted = 0
    for f in files:
        dst_filename = os.path.join(target_dir,
                                    os.path.splitext(os.path.basename(f))[0])

        logging.debug("Converting sample %s, to %s", f, dst_filename)
        ConvertSample(f, dst_filename, show_first_spectrogram)
        show_first_spectrogram = False

        sample_converted += 1
        if sample_converted % 100 == 0:
            logging.info("time elapsed: %d, sample converted: %d",
                         time.time() - start, sample_converted)

    logging.info("Prepare input lasts: %d seconds for %d samples",
                 time.time() - start,
                 len(files))
#
# neural network
#


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactive = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activation', preactive)
        activations = act(preactive, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

#
# Main
#


def MnistNNTrain():
    common.LogLevel(logging.INFO)
    # parameters
    hidden_layer_size = 500
    batch_size = 500
    training_rate = 5e-4
    epoch = 2000

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    logging.debug("x shape = %s, y shape = %s", batch_xs.shape, batch_ys.shape)

    dim = batch_xs.shape[1]
    num_classes = batch_ys.shape[1]
    assert batch_xs.shape[0] == batch_size
    assert batch_ys.shape[0] == batch_size

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, dim])
        y_ = tf.placeholder(tf.float32, [None, num_classes])

    nn_layer1 = \
        nn_layer(x, dim, hidden_layer_size, "nn_layer1")

    nn_layer2 = \
        nn_layer(nn_layer1,
                 hidden_layer_size,
                 num_classes, "nn_layer2", act=tf.identity)

    cross_entropy = \
        tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_,
                logits=nn_layer2))

    train = tf.train.AdamOptimizer(training_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(nn_layer2, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            logging.debug("epoch: %d", i)
            _, train_accuracy = \
                sess.run([train, accuracy],
                         feed_dict={x: batch_xs, y_: batch_ys})
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            logging.info("accuracy: %f", train_accuracy)


def SystemTestNNTrain():
    common.LogLevel(logging.INFO)
    src_dir = os.path.join(common.SrcDir(), "system-sample-2017-11-08")
    input_dir = os.path.join(common.DataDir(), "system-sample-mel-spectrogram")

    if not os.path.exists(input_dir):
        PrepareInput(src_dir, input_dir)
    else:
        logging.info("Skipping preparing input dir %s since it already exists",
                     input_dir)


def main():
    # SystemTestNNTrain()
    MnistNNTrain()

if __name__ == "__main__":
    main()
