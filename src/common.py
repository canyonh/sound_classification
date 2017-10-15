import os


def RootDir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def DataDir():
    return os.path.join(RootDir(), "data")
