import os
import dataset
import common
import model
import logging
from tensorflow.examples.tutorials.mnist import input_data

#
# Tests
#


# Linear model sanity test
def TestLinearModel():
    test_dataset = dataset.NpyDataSet()
    dir_path = os.path.join(common.DataDir(), "systest-prototype-small-npy")
    print(dir_path)
    test_dataset.Load(dir_path)
    with model.LinearModel() as linear_model:
        linear_model.Train(test_dataset, 10, 1)
        first_row = test_dataset.x[0]
        first_row = first_row.reshape(1, first_row.shape[0])
        infer_result = linear_model.Infer(first_row)
        logging.info("infer result: %d", infer_result)


# test with MNIST data
def TestLinearMnistModel():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    training_set, training_label = mnist.train.next_batch(60000)
    mnist_dataset = dataset.DataSet()
    mnist_dataset.Set(training_set, training_label)

    with model.LinearModel() as mdl:
        mdl.Train(mnist_dataset, 100)


# Test with MNIST data
def TestNNModelMnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    training_set, training_label = mnist.train.next_batch(60000)
    mnist_dataset = dataset.DataSet()
    mnist_dataset.Set(training_set, training_label)

    with model.SimpleNeuralNetwork(2, 500) as mdl:
        mdl.Train(mnist_dataset, 100)
        '''
        first_row = mnist_dataset.x[0]
        first_row = first_row.reshape(1, first_row.shape[0])
        infer_result = linear_model.Infer(first_row)
        logging.info("infer result: %d", infer_result)
        '''


def TestSaveModel():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    training_set, training_label = mnist.train.next_batch(60000)
    mnist_dataset = dataset.DataSet()
    mnist_dataset.Set(training_set, training_label)

    with model.SimpleNeuralNetwork(2, 500) as mdl:
        mdl.Train(mnist_dataset, 100)
        mdl.SaveModel("/tmp/nn.ckpt")


def TestLoadModel():
    with model.SimpleNeuralNetwork(2, 500) as mdl:
        mdl.LoadModel("/tmp/nn.ckpt")
