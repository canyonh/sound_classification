import os
import glob
import librosa
import random
import numpy as np
import logging
from random import shuffle

class DataSet:

    def __init__(self):
        # index is numerical class
        self.lables = []
        self.x = []
        self.y = []

    def Load(self, path):
        files = glob.glob(path)
        shuffle(files)
        
        for f in files:
            logging.debug("Load file %s", f)
            label = __CalculateLable(f)
            x.appennd(__LoadImpl(self, f))

        __OneHotEncode(self.y)
        return

    # split the labels, put in labels, and calculate the correct class
    def __CalculateLabel(file_name):
        token = os.path.basename(file_name).split('_')[0]
        logging.debug("assign label %s for file %s", token, file_name)

        idx = -1
        try:
            idx = self.labels.index(token)
        except ValueError:
            break;
        
        if idx == -1:
            self.labels.append(token)
            correct_class = len(self.labels)
            logging.debug("assign existing label %s for file %s, class: %d", token, file_name, correct_class)
        else:
            correct_class = idx
            logging.debug("adding a new label %s for file %s, class: %d", token, file_name, correct_class)
        __LoadImpl(self,file_name)

    def __LoadImpl(self, path):
        raise NotImplementedError()
        return None

    def __OneHotEncoding(dim, val):
        max_class = len(self.labels) + 1
        assert max
        return

class WaveDataSet(DataSet):

    def __LoadImpl(self, path):
        
        


        

class Sample:

    def __init__(self):
        self.path = ""
        self.data = None
        self.correct_class = None
        self.sample_rate = 8000

    def Load(self, path):
        raise NotImplementedError()

    

    def Load(self, path):
        basename = os.path.basename(path)
        tokens = basename.split('.')
        print(tokens)
        assert len(tokens) == 2

        label = __CalculateLabel(name)
        if tokens[1] == 'npy':
            LoadNpySample(tokens[0], label)
        else:
            LoadWavSample(tokens[0], label)
        return


    def LoadNpySample(name):
        sample_rate, npy
        return

    def LoadWavSample(name):
        sample_rate, wav_data = sci.io.wavfile.read(name)
        return

    def ConvertToNpy():
        return

def main():
    print("test\n")
    return

if __name__ == "__main__":
    main()
