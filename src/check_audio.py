import os
import sys
import glob
import librosa
import random
import numpy as np
import logging
from random import shuffle

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
            logging.debug("Loading file %s", f)
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
        return samples
#
# conversion
#

# resample in-place
def Resample(filename, target_sampling_rate):
    y, _ = librosa.load(filename, sr=target_sampling_rate)
    librosa.output.write_wav(filename, y, target_sample_rate)

# generate spectrogram
def GenerateSpectrogram(src_filename, target_filename, sampling_rate, num_fft=320, hop_len=160, num_mels=80, flatten=True)
    logging.debug("GenerateSpectrogram(), src file: %s, dst file: %s", src_filename, target_filename)
    y, _ = librosa.core.load(src_filename, sr=sampling_rate)
    spectro = librosa.feature.melspectrogram(y, sr=sampling_rate, n_fft=num_fft, hop_length=hop_len, n_mels=num_mels)
    if flatten == True:
        np.save(target_sampling_rate, spectro.flatten()))
    else:
        np.save(target_filename, spectro)

def ResampleDir(target_dir):
    files = glob.glob(os.path.join(target_dir, ".*"))
    for f in files:
        logging.debug("resample %s to 8000 HZ", f)
        Resample(f, 8000)

def RootDir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

def ShowLog():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
   
#
# Tests
#
def TestLoadSample():
    dir_path = os.path.join(RootDir(),"data/systest-prototype-small")
    wave_data_set = WaveDataSet()
    wave_data_set.Load(dir_path)
   
def main():
    ShowLog()
    # only need once
    #ResampleDir(os.path.join(RootDir(), "data/systest-prototype-small"))
    TestLoadSample()
    
    return

if __name__ == "__main__":
    main()
