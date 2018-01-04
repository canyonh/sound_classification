# import dataset
import logging
import common
import test


def main():
    common.LogLevel(logging.DEBUG)
    # only need once
    # dataset.TestDataSets()
    # test.TestLinearModel()
    # test.TestLinearMnistModel()
    # test.TestNNModelMnist()
    test.TestSaveModel()
    test.TestLoadModel()

if __name__ == "__main__":
    main()
