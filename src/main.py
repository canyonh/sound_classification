# import dataset
import common
import test


def main():
    common.ShowLog()
    # only need once
    # dataset.TestDataSets()
    # test.TestLinearModel()
    test.TestLinearMnistModel()
    test.TestNNModelMnist()

if __name__ == "__main__":
    main()
