import os
import logging
import sys


def ShowLog():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def RootDir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def DataDir():
    return os.path.join(RootDir(), "data")


def SrcDir():
    return os.path.join(RootDir(), "data-src")
