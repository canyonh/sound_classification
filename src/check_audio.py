import os
import glob
import librosa
import random
import numpy as np
import librosa.display
import time
import scipy.io.wavfile
import math
from random import shuffle

class SampleSet:

    def __init__(self):
        # index is numerical class
        self.lables = []

        # samples and its correct class
        self.x = []
        self.y = []

    def load(directory, 
        

class Sample:

    def __init__(self):
        self.path = ""
        self.data = None
        self.correct_class = None
        self.sample_rate = 8000

    def Load(self, path):
        raise NotImplementedError()

    def __CalculateLabel(file_name):
        #<label>_<serial>.type
        token = os.path.basename(file_name).split('_')[0]
        if token in self.labels:
            correct_class = self.LoadNpySample(file_name)
        else:
            correct_class = len(self.labels)
            labels.append(token)
        return token
    

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
