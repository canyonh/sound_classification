import os
import glob
import librosa
import random
import numpy as np
import logging
from random import shuffle
import scipy.io.wavfile

class DataSet:

    def __init__(self):
        # index is numerical class
        self.labels = []
        self.x = []
        self.y = []

    def Load(self, path):
        files = glob.glob(os.path.join(path, "*"))
        shuffle(files)
        
        for f in files:
            logging.debug("Load file %s", f)
            label = self.__CalculateLabel(f)
            x.appennd(self.LoadImpl(f))

        __OneHotEncode(self.y)
        return

    # split the labels, put in labels, and calculate the correct class
    def __CalculateLabel(self, file_name):
        token = os.path.basename(file_name).split('_')[0]
        logging.debug("assign label %s for file %s", token, file_name)

        idx = -1
        try:
            idx = self.labels.index(token)
        except ValueError:
            pass
        
        if idx == -1:
            self.labels.append(token)
            correct_class = len(self.labels)
            logging.debug("assign existing label %s for file %s, class: %d", token, file_name, correct_class)
        else:
            correct_class = idx
            logging.debug("adding a new label %s for file %s, class: %d", token, file_name, correct_class)

        self.LoadImpl(file_name)

    def LoadImpl(self, path):
        raise NotImplementedError()
        return None

    def __OneHotEncoding(dim, val):
        max_class = len(self.labels) + 1
        assert max
        return

class WaveDataSet(DataSet):

    def LoadImpl(self, path):
        sample_rate, samples = scipy.io.wavfile.read(path)
        assert sample_rate == 8000 # @todo we only test with 8000 now
        return samples
        
def TestLoadSample():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, "..","samples")
    wave_data_set = WaveDataSet()
    wave_data_set.Load(dir_path)
   
def main():
    TestLoadSample()
    return

if __name__ == "__main__":
    main()
