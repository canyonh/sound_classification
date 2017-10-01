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
        
        correct_classes = []
        for f in files:
            logging.debug("Loading file %s", f)
            correct_classes.append(self.__CalculateLabel(f))
            self.x.append(self.LoadImpl(f))

        # covert y to one hot encoding
        for correct_class in correct_classes:
            self.y.append(self.__OneHotEncoding(correct_class))

    # split the labels, put in labels, and calculate the correct class
    def __CalculateLabel(self, file_name):
        base_name = os.path.basename(file_name)
        token = base_name.split('_')[0]
        logging.debug("assign label %s for file %s", token, base_name)

        idx = -1
        try:
            idx = self.labels.index(token)
        except ValueError:
            pass
        
        if idx == -1:
            correct_class = len(self.labels)
            self.labels.append(token)
            logging.debug("assign existing label %s for file %s, class: %d", token, base_name, correct_class)
        else:
            correct_class = idx
            logging.debug("adding a new label %s for file %s, class: %d", token, base_name, correct_class)
        
        return correct_class

    def LoadImpl(self, path):
        raise NotImplementedError()
        return None

    def __OneHotEncoding(self, correct_class):
        num_classes = len(self.labels)
        one_hot = np.zeros(num_classes)
        one_hot[correct_class] = 1
        #print("total: ", num_classes, "class: ", correct_class, "one hot: ", one_hot)
        return one_hot

class WaveDataSet(DataSet):

    def LoadImpl(self, path):
        y, sr= librosa.load(path, 8000)
        logging.debug("wave file %s loaded, sample rate:%d",path, sr)
        return y

#
# conversion
#

# resample in-place
def Resample(filename, target_sampling_rate):
    y, _ = librosa.load(filename, sr=target_sampling_rate)
    librosa.output.write_wav(filename, y, target_sample_rate)

# generate spectrogram
def GenerateSpectrogram(src_filename, target_filename, sampling_rate, num_fft=320, hop_len=160, num_mels=80, flatten=True):
    logging.debug("GenerateSpectrogram(), src file: %s, dst file: %s", src_filename, target_filename)
    y, _ = librosa.core.load(src_filename, sr=sampling_rate)
    spectro = librosa.feature.melspectrogram(y, sr=sampling_rate, n_fft=num_fft, hop_length=hop_len, n_mels=num_mels)
    if flatten == True:
        np.save(target_sampling_rate, spectro.flatten())
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
