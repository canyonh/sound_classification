# import dataset
import logging
import common
# import test
import dataset


def main():
    common.LogLevel(logging.DEBUG)
    # only need once
    # dataset.TestDataSets()
    # test.TestLinearModel()
    # test.TestLinearMnistModel()
    # test.TestNNModelMnist()
    # test.TestSaveModel()
    # test.TestLoadModel()
    dataset.TestLoadSample()

if __name__ == "__main__":
    main()
